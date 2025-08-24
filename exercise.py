# live_side_by_side.py
import argparse
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
from ui_priority import show_priority_ui, show_setup_ui, build_weights_from_priority
from scoring import (
    preprocess_for_gcn,
    embed_sequence,
    dtw_distance_cosine,
    dtw_similarity,
    dtw_distance_l1,
    extract_joint_angles_xy,
    compute_angles_for_seq,
    total_motion_amplitude,
    calibrate_score,
    masked_motion_amplitude,
    build_priority_mask,
    smooth_angles,
    resample_to_length,
)
from orientation import compute_forward_vector_3d, average_forward_vector
from weights_detection import detect_weights
from feedback_system import create_feedback_system

# SimpleGCN no longer used (angle-based scoring)

# ------------------------------ Pose Setup ------------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ------------------------------ Geometry Utils ------------------------------
# Moved to scoring.py

# ------------------------------ MediaPipe helpers ------------------------------
def extract_landmarks(frame_bgr, pose_model):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = pose_model.process(rgb)
    if result.pose_landmarks:
        lms = result.pose_landmarks.landmark
        arr = np.array([(lm.x, lm.y, lm.z) for lm in lms], dtype=np.float32)  # [33,3]
        return result.pose_landmarks, arr
    return None, None

def extract_pose_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    seq = []
    with mp_pose.Pose() as pose_trainer:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            lmk_obj, lmk_arr = extract_landmarks(frame, pose_trainer)
            if lmk_arr is not None and lmk_arr.shape == (33, 3):
                seq.append(lmk_arr)
    cap.release()
    return seq  # list of [33,3]

# ------------------------------ Orientation (3D) ------------------------------
# Moved to orientation.py

# ------------------------------ GCN Preprocess/Embed ------------------------------
# Moved to scoring.py

"""
All detailed scoring and orientation helpers moved to scoring.py and orientation.py
to avoid duplication and keep this file focused on session orchestration.
"""

# ------------------------------ Rep Detection ------------------------------
class RepDetector:
    """
    Simple, robust 1D cycle detector using angle signal.
    - Smooths with moving average
    - State machine with hysteresis around trainer-derived thresholds
    """
    def __init__(self, window=5, min_amp=20.0, hysteresis=5.0):
        self.window = window
        self.buf = deque(maxlen=window)
        self.state = 'idle'  # 'down' or 'up'
        self.last_extreme = None
        self.min_amp = min_amp
        self.hys = hysteresis
        self.rep_segments = []   # list of (start_idx, end_idx)
        self.current_start = None

    def smooth(self, x):
        self.buf.append(x)
        return sum(self.buf) / len(self.buf)

    def update(self, t_idx, angle, low_thresh, high_thresh):
        """
        low_thresh, high_thresh determined from trainer angle distribution.
        """
        val = self.smooth(angle)

        if self.state == 'idle':
            # initialize even if starting between thresholds
            self.current_start = t_idx
            if val >= (low_thresh + high_thresh) * 0.5:
                self.state = 'up'
            else:
                self.state = 'down'
            return None

        if self.state == 'down':
            # wait until enough above (high_thresh + hys) to count a half-cycle
            if val >= (high_thresh + self.hys):
                # completed down->up transition
                self.state = 'up'
                return None

        elif self.state == 'up':
            # wait until below (low_thresh - hys) to count a full cycle (rep)
            if val <= (low_thresh - self.hys):
                # completed up->down, this closes a rep
                if self.current_start is not None:
                    self.rep_segments.append((self.current_start, t_idx))
                self.current_start = t_idx
                self.state = 'down'
                return self.rep_segments[-1]  # recently closed rep

        return None

def derive_angle_thresholds(angle_series):
    """
    Given a clean trainer angle series, compute adaptive thresholds.
    """
    if len(angle_series) == 0:
        return 40.0, 140.0, 20.0, 5.0
    lo = np.percentile(angle_series, 15)
    hi = np.percentile(angle_series, 85)
    amp = max(hi - lo, 20.0)
    hyst = max(0.05 * amp, 5.0)
    return float(lo), float(hi), float(amp), float(hyst)

# ------------------------------ Main Live Session ------------------------------
def run_live_session(trainer_video_path, device='cpu', hidden=64, priority=None, priority_weight=1.5, nonpriority_weight=0.5, require_weights=False):
    # 1) Load trainer sequence
    trainer_seq = extract_pose_sequence(trainer_video_path)
    if len(trainer_seq) == 0:
        print("[ERROR] Could not extract trainer landmarks.")
        return

    # Choose angle channel for rep detection from trainer
    trainer_angles = []
    for fr in trainer_seq:
        a = extract_joint_angles_xy(fr)
        # choose one angle (example: pick the one with max variance across trainer)
        trainer_angles.append([a['elbow_l'], a['elbow_r'], a['knee_l'], a['knee_r']])
    trainer_angles = np.asarray(trainer_angles)  # [T, 4]
    var_by_joint = trainer_angles.var(axis=0)
    chosen_idx = int(np.argmax(var_by_joint))
    chosen_name = ['elbow_l','elbow_r','knee_l','knee_r'][chosen_idx]
    trainer_angle_1d = trainer_angles[:, chosen_idx]

    low_t, high_t, amp, hyst = derive_angle_thresholds(trainer_angle_1d)

    # 2) Build trainer template (angles-based scoring; no GCN needed)

    # Detect rep segments on the trainer angle series to extract a single-rep template
    trainer_repdet = RepDetector(window=5, min_amp=amp, hysteresis=hyst)
    trainer_rep_segments = []
    for t_idx, ang in enumerate(trainer_angle_1d):
        seg = trainer_repdet.update(t_idx, float(ang), low_t, high_t)
        if seg is not None:
            trainer_rep_segments.append(seg)

    # Choose the longest/most stable trainer rep as the template; fallback to full video if none
    if trainer_rep_segments:
        trainer_rep_segments.sort(key=lambda se: se[1] - se[0], reverse=True)
        ts, te = trainer_rep_segments[0]
        ts = max(0, ts)
        te = min(te, len(trainer_seq) - 1)
        # enforce a minimal duration to avoid noise
        if te > ts + 2:
            trainer_template_seq = trainer_seq[ts:te+1]
        else:
            trainer_template_seq = trainer_seq
    else:
        trainer_template_seq = trainer_seq

    # trainer_emb (GCN) not used; kept removed for efficiency

    # Precompute trainer angles (smoothed), and weights for joints
    trainer_angles = compute_angles_for_seq(trainer_template_seq)
    trainer_angles = smooth_angles(trainer_angles, window=5)
    D = trainer_angles.shape[1]
    # Trainer forward orientation (3D) for coarse direction check
    trainer_forward = average_forward_vector(trainer_template_seq)
    # Build weights from user-selected priorities
    default_weights = build_weights_from_priority(priority, priority_weight, nonpriority_weight, D)
    priority_mask = build_priority_mask(priority, D)
    
    # Initialize feedback system
    feedback_system = create_feedback_system(priority, default_weights)

    # 3) Set up live capture
    cap_trainer = cv2.VideoCapture(trainer_video_path)
    cap_user = cv2.VideoCapture(0)
    if not cap_user.isOpened():
        print("[ERROR] Webcam not available.")
        return

    user_pose = mp_pose.Pose()
    trainer_pose = mp_pose.Pose()

    # Buffer of recent user frames so we can slice by trainer rep duration
    trainer_rep_len = max(1, len(trainer_template_seq))
    user_buf = deque(maxlen=int(2 * trainer_rep_len))  # keep ~2 reps worth of frames
    rep_scores = []
    last_rep_score = None

    print(f"[INFO] Live session started. Alignment check in progress...")
    win = "Trainer (Left) | You (Right)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_landmark_time = 0.0

    # Track trainer frame index to detect loop end boundaries
    prev_trainer_frame_idx = int(cap_trainer.get(cv2.CAP_PROP_POS_FRAMES))
    loop_pending = False
    prev_looped = False
    # Alignment gating
    # Alignment/weights gates (continuous)
    align_buf = deque(maxlen=36)
    align_needed_len = min(24, len(trainer_angles))
    trainer_align_ref = trainer_angles[:align_needed_len]
    align_margin_deg = 2.0
    weights_ok = (not require_weights)
    orient_ok = False
    # Pre-start gates: check only before workout starts; require stability over one loop
    pre_start_mode = True
    stable_orient_ok = False
    stable_weights_ok = (not require_weights)
    orientation_locked = False  # Once True, stop checking orientation
    weights_locked = (not require_weights)  # Once True, stop checking weights
    
    # State for showing start message
    show_start_message = False
    start_message_time = 0.0
    
    # Feedback system state
    current_feedback = ""
    feedback_display_time = 0.0
    feedback_duration = 3.0  # Show feedback for 3 seconds

    while True:
        ok_t, trainer_frame = cap_trainer.read()
        ok_u, user_frame = cap_user.read()
        if not ok_u:
            print("[ERROR] Could not read webcam frame.")
            break

        # Detect trainer loop end to trigger one rep evaluation
        trainer_frame_idx = int(cap_trainer.get(cv2.CAP_PROP_POS_FRAMES))
        looped = False
        if not ok_t:
            looped = True
        elif trainer_frame_idx < prev_trainer_frame_idx:
            looped = True
        prev_trainer_frame_idx = trainer_frame_idx if ok_t else prev_trainer_frame_idx

        if not ok_t:
            # Reset to start for next loop
            cap_trainer.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok_t, trainer_frame = cap_trainer.read()
            if not ok_t:
                break
        # Arm scoring only on transition to looped (False -> True)
        if looped and not prev_looped and not loop_pending:
            loop_pending = True

        trainer_frame = cv2.resize(trainer_frame, (640, 480))
        user_frame = cv2.resize(user_frame, (640, 480))
        user_frame = cv2.flip(user_frame, 1)

        current_time = time.time()
        if current_time - last_landmark_time >= 0.04:  # ~25 FPS
            last_landmark_time = current_time
            tr_lmk_obj, _ = extract_landmarks(trainer_frame, trainer_pose)
            us_lmk_obj, us_lmk_arr = extract_landmarks(user_frame, user_pose)

            if us_lmk_arr is not None and us_lmk_arr.shape == (33, 3):
                user_buf.append(us_lmk_arr)
                align_buf.append(us_lmk_arr)

            # Orientation check only before start: compare normal vs mirrored DTW + forward vector
            if pre_start_mode and not orientation_locked and len(align_buf) >= align_needed_len:
                A_us = compute_angles_for_seq(list(align_buf)[-align_needed_len:])
                A_us = smooth_angles(A_us, window=5)
                A_us = resample_to_length(A_us, len(trainer_align_ref))
                dist_nom = dtw_distance_l1(A_us, trainer_align_ref, weights=default_weights)
                # mirrored
                A_us_m = A_us.copy()
                A_us_m[:, [0,1]] = A_us[:, [1,0]]
                A_us_m[:, [2,3]] = A_us[:, [3,2]]
                A_us_m[:, [4,5]] = A_us[:, [5,4]]
                A_us_m[:, [6,7]] = A_us[:, [7,6]]
                dist_mir = dtw_distance_l1(A_us_m, trainer_align_ref, weights=default_weights)
                orient_ok = ((dist_nom <= dist_mir * 0.99) or (dist_nom + align_margin_deg <= dist_mir))
                user_forward = average_forward_vector(list(align_buf)[-align_needed_len:])
                if trainer_forward is not None and user_forward is not None:
                    cos_dir = float(np.clip(np.dot(trainer_forward, user_forward), -1.0, 1.0))
                    if cos_dir < 0.5:
                        orient_ok = False

            # Weights detection delegated to weights_detection.detect_weights
            if pre_start_mode and not weights_locked and require_weights and us_lmk_arr is not None:
                if 'wrist_hist' not in locals():
                    wrist_hist = deque(maxlen=45)
                wrist_hist.append(us_lmk_arr[[15,16,11,12], :2])  # x,y only
                weights_ok = detect_weights(wrist_hist)
                if weights_ok:
                    print(f"[DEBUG] Weights detected! Buffer size: {len(wrist_hist)}")
            # On loop boundary, update stable gates based on the last window
            if pre_start_mode and loop_pending:
                # Lock orientation if it's been stable for one loop
                if orient_ok and not orientation_locked:
                    orientation_locked = True
                    stable_orient_ok = True
                    print(f"[INFO] Orientation locked! Orientation: {stable_orient_ok}")
                elif not orientation_locked:
                    stable_orient_ok = orient_ok
                
                # Lock weights if they've been detected for one loop
                if weights_ok and not weights_locked:
                    weights_locked = True
                    stable_weights_ok = True
                    print(f"[INFO] Weights locked! Weights: {stable_weights_ok}")
                elif not weights_locked:
                    stable_weights_ok = (not require_weights) or weights_ok
                
                # Debug output
                print(f"[DEBUG] Loop boundary - Orientation: {stable_orient_ok} (locked: {orientation_locked}), Weights: {stable_weights_ok} (locked: {weights_locked}), require_weights: {require_weights}")

            # Start workout after orientation is aligned (and weights if required) on a loop boundary
            if pre_start_mode and stable_orient_ok and ((not require_weights) or stable_weights_ok) and loop_pending:
                # Set flag to show start message for 2 seconds
                if require_weights:
                    print("[INFO] Both orientation and weights conditions met! Starting exercise in 2 seconds...")
                else:
                    print("[INFO] Orientation aligned! Starting exercise in 2 seconds...")
                show_start_message = True
                start_message_time = time.time()
                loop_pending = False
            # Check if start message time has elapsed and start workout
            if show_start_message and (time.time() - start_message_time) >= 2.0:
                print("[INFO] Starting workout now!")
                pre_start_mode = False
                show_start_message = False
                # reset for clean start
                user_buf.clear()
                if 'wrist_hist' in locals():
                    wrist_hist.clear()
                loop_pending = False
                prev_looped = False
                cap_trainer.set(cv2.CAP_PROP_POS_FRAMES, 0)
                prev_trainer_frame_idx = 0
            
            # Score only after start
            if (not pre_start_mode) and loop_pending:
                # Take the most recent trainer_rep_len frames; if fewer, use whatever is available
                if len(user_buf) == 0:
                    # No user data; assign worst possible similarity (0.0)
                    score = 0.0
                else:
                    user_segment = list(user_buf)[-trainer_rep_len:]
                    try:
                        # Angle-based DTW with amplitude penalty for robustness
                        A_tr = trainer_angles  # precomputed
                        A_us = compute_angles_for_seq(user_segment)
                        # optional auto-mirroring: flip left/right if mirroring yields lower DTW
                        A_us_sm = smooth_angles(A_us, window=5)
                        A_us_rs = resample_to_length(A_us_sm, len(A_tr))
                        dist_nom = dtw_distance_l1(A_us_rs, A_tr, weights=default_weights)
                        # mirror left/right columns: (0,1), (2,3), (4,5), (6,7)
                        A_us_mir = A_us_rs.copy()
                        A_us_mir[:, [0,1]] = A_us_rs[:, [1,0]]
                        A_us_mir[:, [2,3]] = A_us_rs[:, [3,2]]
                        A_us_mir[:, [4,5]] = A_us_rs[:, [5,4]]
                        A_us_mir[:, [6,7]] = A_us_rs[:, [7,6]]
                        dist_mir = dtw_distance_l1(A_us_mir, A_tr, weights=default_weights)
                        dist = min(dist_nom, dist_mir)
                        sim_angle = np.exp(-0.03 * dist)
                        # amplitude computed over priority joints only
                        amp_user_pr = masked_motion_amplitude(A_us_rs, priority_mask)
                        amp_tr_pr = masked_motion_amplitude(A_tr, priority_mask) + 1e-6
                        amp_ratio = float(np.clip(amp_user_pr / amp_tr_pr, 0.0, 1.0))
                        # First calculate the base score based on priority joint performance
                        base_score = float(sim_angle * amp_ratio)
                        
                        # Only check non-priority motion if priority motion is good enough
                        if base_score >= 0.4:  # If priority joints are performing well
                            # Check non-priority joints only when priority motion is good
                            non_mask = ~priority_mask if np.any(priority_mask) else np.zeros_like(priority_mask)
                            amp_user_np = masked_motion_amplitude(A_us_rs, non_mask)
                            amp_trainer_np = masked_motion_amplitude(A_tr, non_mask)
                            
                            # Calculate how much user's non-priority motion differs from trainer
                            np_motion_ratio = amp_user_np / (amp_trainer_np + 1e-6)
                            
                            # Only penalize if user is moving non-priority joints MUCH more than trainer
                            if np_motion_ratio > 2.5:  # User moving 150% more than trainer
                                excessive_motion_penalty = 0.3  # Heavy penalty for wrong body part motion
                                score = base_score * excessive_motion_penalty
                                print(f"[DEBUG] Excessive non-priority motion detected: user={amp_user_np:.3f}, trainer={amp_trainer_np:.3f}, ratio={np_motion_ratio:.2f}")
                                print(f"[SCORING] Applying penalty: {excessive_motion_penalty}, final score: {score:.3f}")
                            else:
                                # Non-priority motion is reasonable, keep good score
                                score = base_score
                                print(f"[DEBUG] Non-priority motion OK: user={amp_user_np:.3f}, trainer={amp_trainer_np:.3f}, ratio={np_motion_ratio:.2f}")
                                print(f"[SCORING] Keeping good score: {score:.3f}")
                        else:
                            # Priority motion is poor, don't worry about non-priority
                            score = base_score
                            print(f"[DEBUG] Priority motion poor ({base_score:.3f}), skipping non-priority check")
                        
                        # apply a gentle nonlinear boost to encourage near-correct reps
                        score = calibrate_score(score, gamma=0.8)
                    except Exception as e:
                        print(f"[DEBUG] Scoring error: {e}")
                        score = 0.0
                rep_scores.append(score)
                last_rep_score = score
                
                # Generate real-time feedback for this rep
                if len(user_buf) > 0:
                    user_segment_angles = compute_angles_for_seq(user_segment)
                    user_motion_amp = masked_motion_amplitude(user_segment_angles, priority_mask)
                    trainer_motion_amp = masked_motion_amplitude(A_tr, priority_mask)
                    
                    # Get feedback from the system
                    feedback = feedback_system.analyze_rep_performance(
                        user_segment_angles, A_tr, user_motion_amp, 
                        trainer_motion_amp, score, priority_mask
                    )
                    
                    # Update feedback display
                    current_feedback = feedback
                    feedback_display_time = time.time()
                    print(f"[FEEDBACK] {feedback}")
                
                loop_pending = False
            
            # Continuous feedback during exercise (not just after reps)
            elif (not pre_start_mode) and len(user_buf) >= 10:
                # Check if user is moving enough
                recent_user_angles = compute_angles_for_seq(list(user_buf)[-10:])
                user_motion_amp = masked_motion_amplitude(recent_user_angles, priority_mask)
                
                # Only show continuous feedback if no rep feedback is currently displayed
                if not current_feedback or (time.time() - feedback_display_time) >= feedback_duration:
                    if user_motion_amp < 0.02:  # Very low motion
                        current_feedback = "Start moving! Follow the trainer"
                        feedback_display_time = time.time()
                    elif user_motion_amp < 0.05:  # Low motion
                        current_feedback = "Move more! Increase your range"
                        feedback_display_time = time.time()

        # Track loop state for next iteration to avoid double-arming
        prev_looped = looped

            # draw landmarks
        if tr_lmk_obj:
            mp_drawing.draw_landmarks(trainer_frame, tr_lmk_obj, mp_pose.POSE_CONNECTIONS)
        if us_lmk_obj:
            mp_drawing.draw_landmarks(user_frame, us_lmk_obj, mp_pose.POSE_CONNECTIONS)

        combined = np.hstack((trainer_frame, user_frame))
        cv2.putText(combined, f"Reps (trainer-driven): {len(rep_scores)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        if last_rep_score is not None:
            cv2.putText(combined, f"Last Rep Score: {last_rep_score:.3f}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 255, 50), 2)
        
        # Display real-time feedback
        if current_feedback and (time.time() - feedback_display_time) < feedback_duration:
            # Choose color based on feedback type
            if "Excellent" in current_feedback or "Good" in current_feedback:
                color = (50, 255, 50)  # Green for positive feedback
            elif "Start moving" in current_feedback:
                color = (0, 200, 255)  # Orange for movement prompts
            else:
                color = (0, 255, 255)  # Yellow for improvement tips
            
            # Display feedback on screen
            cv2.putText(combined, current_feedback, (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            # Clear feedback after duration expires
            current_feedback = ""
        if pre_start_mode:
            # Use stable (debounced) gate states for UI to avoid flicker
            weights_need_ok = (not require_weights) or stable_weights_ok
            
            # Check if ready to start
            if require_weights:
                # Weights mode: need both orientation and weights
                ready_to_start = stable_orient_ok and stable_weights_ok
            else:
                # No weights mode: only need orientation
                ready_to_start = stable_orient_ok
            
            if ready_to_start:
                # Ready to start - show start message with countdown
                remaining_time = max(0, 2 - int(time.time() - start_message_time)) if 'start_message_time' in locals() else 2
                cv2.putText(combined, "Start exercise!", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 255, 50), 2)
                cv2.putText(combined, f"Starting in {remaining_time} seconds...", (20, 175),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
            else:
                # Show individual status messages
                if not stable_orient_ok:
                    cv2.putText(combined, "Align with trainer orientation...", (20, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 255), 2)
                else:
                    cv2.putText(combined, "Orientation aligned", (20, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 255, 50), 2)
                
                # Only show weights message if weights are required
                if require_weights:
                    if not stable_weights_ok:
                        cv2.putText(combined, "Expecting weights...", (20, 175),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                    else:
                        cv2.putText(combined, "Weights equipped", (20, 175),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
        else:
            # Workout running; no more gate messages
            cv2.putText(combined, "Press 'q' to Quit", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow(win, combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_trainer.release()
    cap_user.release()
    cv2.destroyAllWindows()

    print("\n[INFO] Session ended.")
    print(f"Total Reps: {len(rep_scores)}")
    if rep_scores:
        print("Rep Scores:", [f"{s:.3f}" for s in rep_scores])

# UI helpers moved to ui_priority.py

# ------------------------------ CLI ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_video", type=str, required=True,
                        help="Path to a single-rep trainer video (or short template).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"],
                        help="Torch device (use 'cuda' if you have a GPU).")
    parser.add_argument("--hidden", type=int, default=64, help="GCN hidden size.")
    parser.add_argument("--priority", type=str, default=None,
                        help="Comma-separated joints to prioritize: elbow,shoulder,hip,knee or side-specific like elbow_l")
    parser.add_argument("--priority_weight", type=float, default=1.8,
                        help="Weight for prioritized joints (>= nonpriority_weight).")
    parser.add_argument("--nonpriority_weight", type=float, default=0.2,
                        help="Weight for non-priority joints (0 to de-emphasize).")
    args = parser.parse_args()
    # If no priority provided, open setup UI (priorities + weights mode)
    weights_mode = "without"
    if not args.priority:
        priority, weights_mode = show_setup_ui()
    else:
        priority = [p.strip() for p in args.priority.split(',')] if args.priority else None
    run_live_session(
        args.trainer_video,
        device=args.device,
        hidden=args.hidden,
        priority=priority,
        priority_weight=args.priority_weight,
        nonpriority_weight=args.nonpriority_weight,
        require_weights=(weights_mode == "with"),
    )

