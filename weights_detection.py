import numpy as np
from collections import deque


def detect_weights(wrist_history_xy: deque) -> bool:
    """
    Improved weights detection using multiple heuristics:
    1. Wrist position stability (less jitter when holding weights)
    2. Wrist-to-shoulder distance patterns (more stable when holding objects)
    3. Motion consistency (smoother movement with weights)
    """
    if wrist_history_xy is None or len(wrist_history_xy) < 20:
        return False
    
    try:
        W = np.stack(wrist_history_xy, axis=0)  # [T, 4, 2]
        T = len(W)
        
        # 1. Wrist position stability (less variance when holding weights)
        wrist_l_pos = W[:, 0, :]  # left wrist
        wrist_r_pos = W[:, 1, :]  # right wrist
        
        # Calculate position variance over time
        wrist_l_var = np.var(wrist_l_pos, axis=0).sum()
        wrist_r_var = np.var(wrist_r_pos, axis=0).sum()
        
        # 2. Wrist-to-shoulder distance stability
        shoulder_l = W[:, 2, :]  # left shoulder
        shoulder_r = W[:, 3, :]  # right shoulder
        
        dist_l = np.linalg.norm(wrist_l_pos - shoulder_l, axis=1)
        dist_r = np.linalg.norm(wrist_r_pos - shoulder_r, axis=1)
        
        # Distance should be more stable when holding weights
        dist_l_var = np.var(dist_l)
        dist_r_var = np.var(dist_r)
        
        # 3. Motion smoothness (less jerky when holding weights)
        # Calculate velocity between consecutive frames
        vel_l = np.diff(wrist_l_pos, axis=0)
        vel_r = np.diff(wrist_r_pos, axis=0)
        
        # Velocity magnitude variance (should be lower when holding weights)
        vel_l_mag = np.linalg.norm(vel_l, axis=1)
        vel_r_mag = np.linalg.norm(vel_r, axis=1)
        vel_l_var = np.var(vel_l_mag)
        vel_r_var = np.var(vel_r_mag)
        
        # 4. Check if wrists are in reasonable positions (not too high/low/extreme)
        # Wrists should typically be around shoulder level or slightly below
        wrist_l_y = wrist_l_pos[:, 1]  # y-coordinate (higher = lower on screen)
        wrist_r_y = wrist_r_pos[:, 1]
        shoulder_l_y = shoulder_l[:, 1]
        shoulder_r_y = shoulder_r[:, 1]
        
        # Check if wrists are in reasonable vertical range relative to shoulders
        y_range_ok = (
            np.all(wrist_l_y > shoulder_l_y - 0.3) and  # not too high
            np.all(wrist_l_y < shoulder_l_y + 0.4) and  # not too low
            np.all(wrist_r_y > shoulder_r_y - 0.3) and
            np.all(wrist_r_y < shoulder_r_y + 0.4)
        )
        
        # 5. Combined detection criteria
        # Position variance should be low (stable hands)
        pos_stable = (wrist_l_var < 0.01) and (wrist_r_var < 0.01)
        
        # Distance to shoulders should be stable
        dist_stable = (dist_l_var < 0.015) and (dist_r_var < 0.015)
        
        # Motion should be smooth (not jerky)
        motion_smooth = (vel_l_var < 0.008) and (vel_r_var < 0.008)
        
        # Wrists should be in reasonable positions
        position_reasonable = y_range_ok
        
        # All conditions must be met
        weights_detected = (
            pos_stable and 
            dist_stable and 
            motion_smooth and 
            position_reasonable
        )
        
        return bool(weights_detected)
        
    except Exception as e:
        print(f"[DEBUG] Weights detection error: {e}")
        return False


