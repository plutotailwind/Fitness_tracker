import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import cv2
import pyttsx3
import time
import threading

class ExerciseFeedbackSystem:
    """
    Real-time feedback system that analyzes user performance and provides
    specific, actionable feedback based on joint priorities and trainer motion.
    """
    
    def __init__(self, priority_joints: List[str], priority_weights: np.ndarray):
        self.priority_joints = priority_joints
        self.priority_weights = priority_weights
        self.joint_names = {
            'elbow_l': 'left elbow', 'elbow_r': 'right elbow',
            'shoulder_l': 'left shoulder', 'shoulder_r': 'right shoulder',
            'hip_l': 'left hip', 'hip_r': 'right hip',
            'knee_l': 'left knee', 'knee_r': 'right knee',
            'ankle_l': 'left ankle', 'ankle_r': 'right ankle'
        }
        
        # Feedback thresholds
        self.motion_threshold = 0.02  # Minimum motion to consider "moving"
        self.good_rep_threshold = 0.5
        self.excellent_rep_threshold = 0.8
        
        # Store previous feedback to avoid spam
        self.last_feedback = ""
        self.feedback_cooldown = 0
        self.feedback_history = deque(maxlen=10)
        
        # Voice feedback setup
        self.voice_enabled = True
        self.voice_cooldown = 0.1  # Very short cooldown to allow feedback for every rep
        self.last_voice_time = 0
    
    def _speak_in_thread(self, feedback: str):
        """Speak feedback in a separate thread with its own voice engine."""
        try:
            # Create a fresh voice engine for this feedback
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.8)
            
            print(f"[VOICE] Speaking: {feedback}")
            engine.say(feedback)
            engine.runAndWait()
            engine.stop()
            
            print(f"[VOICE] Finished speaking: {feedback}")
        except Exception as e:
            print(f"[VOICE] Error speaking: {e}")
        finally:
            # Clean up
            try:
                del engine
            except:
                pass

    def speak_feedback(self, feedback: str):
        """Speak feedback in a new thread (non-blocking)."""
        if not self.voice_enabled:
            print(f"[VOICE] Voice disabled. Voice enabled: {self.voice_enabled}")
            return
        
        current_time = time.time()
        # Only check cooldown, allow same feedback to be spoken multiple times
        if current_time - self.last_voice_time < self.voice_cooldown:
            print(f"[VOICE] Skipping feedback due to cooldown: {feedback}")
            return
        
        self.last_voice_time = current_time
        print(f"[VOICE] Starting new thread for: {feedback}")
        
        # Spawn a new thread for this feedback
        feedback_thread = threading.Thread(
            target=self._speak_in_thread, 
            args=(feedback,), 
            daemon=True
        )
        feedback_thread.start()
    
    def analyze_rep_performance(self, 
                               user_angles: np.ndarray, 
                               trainer_angles: np.ndarray,
                               user_motion_amp: float,
                               trainer_motion_amp: float,
                               score: float,
                               priority_mask: np.ndarray) -> str:
        """
        Analyze user performance and generate specific feedback.
        
        Args:
            user_angles: User's joint angles [T, D]
            trainer_angles: Trainer's joint angles [T, D]
            user_motion_amp: User's motion amplitude
            trainer_motion_amp: Trainer's motion amplitude
            score: Rep score (0-1)
            priority_mask: Boolean mask for priority joints
            
        Returns:
            Feedback string
        """
        
        # Check if this is a good rep first
        if score >= self.excellent_rep_threshold:
            feedback = "Excellent rep! Perfect form!"
            self.speak_feedback(feedback)
            return feedback
        elif score >= self.good_rep_threshold:
            feedback = "Good rep! Keep it up!"
            self.speak_feedback(feedback)
            return feedback
        
        # Analyze specific issues
        feedback_parts = []
        
        # 1. Check if user is moving at all
        if user_motion_amp < self.motion_threshold:
            feedback = "Start moving! Follow the trainer's motion"
            self.speak_feedback(feedback)
            return feedback
        
        # 2. Check if user is moving wrong body parts (non-priority joints)
        # Only check this if user is moving enough overall to avoid false positives
        if user_motion_amp >= self.motion_threshold * 2:  # User is moving reasonably
            wrong_motion_feedback = self._analyze_wrong_body_part_motion(
                user_angles, trainer_angles, priority_mask
            )
            if wrong_motion_feedback:
                self.speak_feedback(wrong_motion_feedback)
                return wrong_motion_feedback
        # If user is not moving enough, skip wrong body part check and go to motion amplitude
        
        # 3. Analyze motion amplitude issues
        amp_feedback = self._analyze_motion_amplitude(
            user_motion_amp, trainer_motion_amp, priority_mask
        )
        if amp_feedback:
            feedback_parts.append(amp_feedback)
        
        # 4. Analyze specific joint issues
        joint_feedback = self._analyze_joint_movement(
            user_angles, trainer_angles, priority_mask
        )
        if joint_feedback:
            feedback_parts.append(joint_feedback)
        
        # 5. Analyze timing/rhythm issues
        timing_feedback = self._analyze_timing(user_angles, trainer_angles)
        if timing_feedback:
            feedback_parts.append(timing_feedback)
        
        # Combine feedback
        if feedback_parts:
            # Take the most important feedback (first one)
            feedback = feedback_parts[0]
            self.speak_feedback(feedback)
            return feedback
        else:
            feedback = "Keep going! You're doing well"
            return feedback
    
    def _analyze_motion_amplitude(self, 
                                 user_amp: float, 
                                 trainer_amp: float, 
                                 priority_mask: np.ndarray) -> Optional[str]:
        """Analyze if user is moving enough in priority joints."""
        if user_amp < trainer_amp * 0.3:
            return "Move more! Increase your range of motion"
        elif user_amp < trainer_amp * 0.6:
            return "Move further! Extend your range"
        return None
    
    def _analyze_joint_movement(self, 
                               user_angles: np.ndarray, 
                               trainer_angles: np.ndarray, 
                               priority_mask: np.ndarray) -> Optional[str]:
        """Analyze specific joint movement issues."""
        if user_angles.shape != trainer_angles.shape:
            return None
            
        # Focus on priority joints
        priority_indices = np.where(priority_mask)[0]
        if len(priority_indices) == 0:
            return None
        
        # Calculate angle differences for priority joints
        angle_diffs = np.abs(user_angles - trainer_angles)
        max_diff_idx = np.argmax(np.mean(angle_diffs, axis=0))
        
        # Map index to joint name
        joint_name = self._get_joint_name_by_index(max_diff_idx)
        if not joint_name:
            return None
        
        # Analyze the specific joint
        user_joint_angles = user_angles[:, max_diff_idx]
        trainer_joint_angles = trainer_angles[:, max_diff_idx]
        
        # Check if user is moving this joint enough
        user_range = np.max(user_joint_angles) - np.min(user_joint_angles)
        trainer_range = np.max(trainer_joint_angles) - np.min(trainer_joint_angles)
        
        if user_range < trainer_range * 0.4:
            return f"Move your {joint_name} more!"
        elif user_range < trainer_range * 0.7:
            return f"Extend your {joint_name} further!"
        
        # Check for specific joint issues
        return self._get_specific_joint_feedback(joint_name, user_joint_angles, trainer_joint_angles)
    
    def _get_specific_joint_feedback(self, 
                                   joint_name: str, 
                                   user_angles: np.ndarray, 
                                   trainer_angles: np.ndarray) -> str:
        """Get specific feedback for individual joints."""
        user_avg = np.mean(user_angles)
        trainer_avg = np.mean(trainer_angles)
        
        if 'elbow' in joint_name:
            if user_avg > trainer_avg + 20:
                return f"Straighten your {joint_name} more!"
            elif user_avg < trainer_avg - 20:
                return f"Bend your {joint_name} more!"
        elif 'knee' in joint_name:
            if user_avg > trainer_avg + 15:
                return f"Bend your {joint_name} more!"
            elif user_avg < trainer_avg - 15:
                return f"Straighten your {joint_name} more!"
        elif 'shoulder' in joint_name:
            if user_avg > trainer_avg + 20:
                return f"Lower your {joint_name} more!"
            elif user_avg < trainer_avg - 20:
                return f"Raise your {joint_name} more!"
        elif 'hip' in joint_name:
            if user_avg > trainer_avg + 20:
                return f"Lower your {joint_name} more!"
            elif user_avg < trainer_avg - 20:
                return f"Raise your {joint_name} more!"
        
        return f"Adjust your {joint_name} position!"
    
    def _analyze_wrong_body_part_motion(self, 
                                       user_angles: np.ndarray, 
                                       trainer_angles: np.ndarray, 
                                       priority_mask: np.ndarray) -> Optional[str]:
        """
        Detect when user is moving non-priority joints too much.
        This catches cases like moving legs during arm exercises.
        Only triggers when non-priority joints are moving vigorously.
        """
        if user_angles.shape != trainer_angles.shape:
            return None
        
        # Get non-priority joint indices (joints that shouldn't move much)
        non_priority_indices = np.where(~priority_mask)[0]
        if len(non_priority_indices) == 0:
            return None
        
        # Calculate motion amplitude for each non-priority joint
        non_priority_motion = []
        for idx in non_priority_indices:
            joint_angles = user_angles[:, idx]
            motion_amp = np.max(joint_angles) - np.min(joint_angles)
            non_priority_motion.append((idx, motion_amp))
        
        # Sort by motion amplitude (highest first)
        non_priority_motion.sort(key=lambda x: x[1], reverse=True)
        
        # Only trigger if the highest non-priority motion is significantly high
        # This prevents false positives when user is just not moving much overall
        if len(non_priority_motion) == 0:
            return None
            
        highest_non_priority_motion = non_priority_motion[0][1]
        
        # Check if any non-priority joint is moving vigorously
        # Use a higher threshold to only catch vigorous movement
        vigorous_motion_threshold = 0.25  # Increased threshold to avoid false positives
        
        for idx, motion_amp in non_priority_motion:
            if motion_amp > vigorous_motion_threshold:
                joint_name = self._get_joint_name_by_index(idx)
                if joint_name:
                    # Check if this joint is moving significantly more than trainer
                    trainer_joint_angles = trainer_angles[:, idx]
                    trainer_motion_amp = np.max(trainer_joint_angles) - np.min(trainer_joint_angles)
                    
                    # Only trigger if user is moving this joint MUCH more than trainer
                    # This catches cases like vigorously moving legs during arm exercises
                    if motion_amp > trainer_motion_amp * 2.0:  # User moving 100% more than trainer
                        return f"Stop moving your {joint_name}! Focus on the exercise"
        
        return None
    
    def _analyze_timing(self, 
                        user_angles: np.ndarray, 
                        trainer_angles: np.ndarray) -> Optional[str]:
        """Analyze timing/rhythm issues."""
        if user_angles.shape != trainer_angles.shape:
            return None
        
        # Simple timing analysis - check if user is too fast/slow
        # This is a basic implementation - could be enhanced with DTW analysis
        return None
    
    def _get_joint_name_by_index(self, index: int) -> Optional[str]:
        """Map joint index to joint name."""
        # This mapping should match the order in compute_angles_for_seq
        joint_order = [
            'elbow_l', 'elbow_r', 'shoulder_l', 'shoulder_r',
            'hip_l', 'hip_r', 'knee_l', 'knee_r'
        ]
        
        if 0 <= index < len(joint_order):
            return joint_order[index]
        return None
    
    def get_encouragement_feedback(self, score: float) -> str:
        """Get encouraging feedback based on score."""
        if score >= 0.9:
            return "Perfect! You're a natural!"
        elif score >= 0.8:
            return "Amazing form! Keep it up!"
        elif score >= 0.7:
            return "Great job! You're improving!"
        elif score >= 0.6:
            return "Good work! Almost there!"
        elif score >= 0.5:
            return "Not bad! Keep practicing!"
        elif score >= 0.3:
            return "Keep trying! You'll get it!"
        else:
            return "Don't give up! Practice makes perfect!"
    
    def get_motion_direction_feedback(self, 
                                    user_angles: np.ndarray, 
                                    trainer_angles: np.ndarray,
                                    priority_mask: np.ndarray) -> str:
        """Get feedback about which direction to move."""
        if user_angles.shape != trainer_angles.shape:
            return "Follow the trainer's motion!"
        
        # Find the joint with biggest difference
        angle_diffs = np.abs(user_angles - trainer_angles)
        max_diff_idx = np.argmax(np.mean(angle_diffs, axis=0))
        
        joint_name = self._get_joint_name_by_index(max_diff_idx)
        if not joint_name:
            return "Follow the trainer's motion!"
        
        user_avg = np.mean(user_angles[:, max_diff_idx])
        trainer_avg = np.mean(trainer_angles[:, max_diff_idx])
        
        if user_avg > trainer_avg:
            return f"Lower your {joint_name} more!"
        else:
            return f"Raise your {joint_name} more!"
    
    def format_feedback_for_display(self, feedback: str, score: Optional[float] = None) -> str:
        """Format feedback for on-screen display."""
        if score is not None:
            if score >= self.good_rep_threshold:
                return f"Target: {feedback}"
            else:
                return f"Tip: {feedback}"
        return feedback

def create_feedback_system(priority_joints: List[str], 
                          priority_weights: np.ndarray) -> ExerciseFeedbackSystem:
    """Factory function to create feedback system."""
    return ExerciseFeedbackSystem(priority_joints, priority_weights)
