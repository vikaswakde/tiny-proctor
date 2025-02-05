import cv2
import mediapipe as mp
import numpy as np
import math

class HeadPoseDetector:
    def __init__(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # MediaPipe's key face landmarks for pose estimation
        self.NOSE_TIP = 1
        self.CHIN = 152
        self.LEFT_EYE = 33
        self.RIGHT_EYE = 263
        self.LEFT_MOUTH = 57
        self.RIGHT_MOUTH = 287
        
        # Additional landmarks for better pose estimation
        self.FOREHEAD = 10
        self.NOSE_BRIDGE = 168
        
        # Updated thresholds based on collected data
        self.PITCH_RANGES = {
            'STRAIGHT': (80, 90),    # Looking straight
            'PARTIAL_DOWN': (71, 80), # Partially looking down
            'FULL_DOWN': (54, 70),    # Fully looking down
        }
        
        self.YAW_RANGES = {
            'STRAIGHT': (115, 125),   # Looking straight
            'PARTIAL_DOWN': (95, 105), # Partially looking down
            'FULL_DOWN': (80, 90),     # Fully looking down
            'LEFT': (110, 115),        # Looking left
            'RIGHT': (115, 120)        # Looking right
        }
        
        self.ROLL_RANGES = {
            'STRAIGHT': (-5, 0),      # Looking straight
            'PARTIAL_DOWN': (-3, 0),  # Partially looking down
            'FULL_DOWN': (0, 2),      # Fully looking down
            'LEFT': (-12, -8),        # Looking left
            'RIGHT': (0, 2)           # Looking right
        }
        
        # Confidence thresholds
        self.CONFIDENCE_THRESHOLD = 2  # Minimum confidence needed for detection
        
        # History for temporal smoothing
        self.history_size = 5
        self.pose_history = []

    def get_pose(self, frame):
        """
        Estimate head pose using MediaPipe Face Mesh
        
        Args:
            frame: Input image
            
        Returns:
            angles: (pitch, yaw, roll) in degrees
            status: Head pose status message
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None, "No face detected"

        face_landmarks = results.multi_face_landmarks[0]
        
        # Get image dimensions
        h, w = frame.shape[:2]
        
        # Extract key 3D landmarks
        nose_tip = self._get_landmark_pos(face_landmarks, self.NOSE_TIP, w, h)
        nose_bridge = self._get_landmark_pos(face_landmarks, self.NOSE_BRIDGE, w, h)
        forehead = self._get_landmark_pos(face_landmarks, self.FOREHEAD, w, h)
        chin = self._get_landmark_pos(face_landmarks, self.CHIN, w, h)
        
        # Calculate angles
        pitch = self._calculate_vertical_angle(nose_bridge, chin, forehead)
        yaw = self._calculate_horizontal_angle(nose_tip, nose_bridge)
        roll = self._calculate_roll(face_landmarks, w, h)
        
        angles = np.array([pitch, yaw, roll])
        status = self._get_pose_status(angles)
        
        return angles, status

    def _get_landmark_pos(self, landmarks, landmark_idx, width, height):
        """Convert landmark to 3D coordinates"""
        landmark = landmarks.landmark[landmark_idx]
        return np.array([
            landmark.x * width,
            landmark.y * height,
            landmark.z * width  # Use width for z to maintain aspect ratio
        ])

    def _calculate_vertical_angle(self, nose_bridge, chin, forehead):
        """Calculate vertical head angle (pitch)"""
        vertical_vector = forehead - chin
        reference_vector = np.array([0, 1, 0])
        angle = np.arctan2(
            np.linalg.norm(np.cross(vertical_vector, reference_vector)),
            np.dot(vertical_vector, reference_vector)
        )
        return np.degrees(angle) - 90  # Adjust to make forward-facing 0 degrees

    def _calculate_horizontal_angle(self, nose_tip, nose_bridge):
        """Calculate horizontal head angle (yaw)"""
        nose_vector = nose_tip - nose_bridge
        reference_vector = np.array([0, 0, 1])
        angle = np.arctan2(
            np.linalg.norm(np.cross(nose_vector, reference_vector)),
            np.dot(nose_vector, reference_vector)
        )
        return np.degrees(angle)

    def _calculate_roll(self, landmarks, width, height):
        """Calculate head tilt angle (roll)"""
        left_eye = self._get_landmark_pos(landmarks, self.LEFT_EYE, width, height)
        right_eye = self._get_landmark_pos(landmarks, self.RIGHT_EYE, width, height)
        
        eye_vector = right_eye - left_eye
        reference_vector = np.array([1, 0, 0])
        
        angle = np.arctan2(eye_vector[1], eye_vector[0])
        return np.degrees(angle)

    def _get_pose_status(self, angles):
        """
        Enhanced pose status detection using collected data ranges
        """
        pitch, yaw, roll = angles
        confidence_score = 0
        current_state = "Forward"
        
        # Initialize confidence scores for each state
        state_confidence = {
            "Forward": 0,
            "Partial Down": 0,
            "Looking Down": 0,
            "Looking Left": 0,
            "Looking Right": 0
        }
        
        # Check pitch ranges
        if self.PITCH_RANGES['STRAIGHT'][0] <= pitch <= self.PITCH_RANGES['STRAIGHT'][1]:
            state_confidence["Forward"] += 2
        elif self.PITCH_RANGES['PARTIAL_DOWN'][0] <= pitch <= self.PITCH_RANGES['PARTIAL_DOWN'][1]:
            state_confidence["Partial Down"] += 2
        elif self.PITCH_RANGES['FULL_DOWN'][0] <= pitch <= self.PITCH_RANGES['FULL_DOWN'][1]:
            state_confidence["Looking Down"] += 2
            
        # Check yaw ranges
        if self.YAW_RANGES['STRAIGHT'][0] <= yaw <= self.YAW_RANGES['STRAIGHT'][1]:
            state_confidence["Forward"] += 1
        elif yaw < self.YAW_RANGES['FULL_DOWN'][1]:
            state_confidence["Looking Down"] += 1
        elif self.YAW_RANGES['LEFT'][0] <= yaw <= self.YAW_RANGES['LEFT'][1]:
            state_confidence["Looking Left"] += 2
        elif self.YAW_RANGES['RIGHT'][0] <= yaw <= self.YAW_RANGES['RIGHT'][1]:
            state_confidence["Looking Right"] += 2
            
        # Check roll ranges for additional confidence
        if self.ROLL_RANGES['STRAIGHT'][0] <= roll <= self.ROLL_RANGES['STRAIGHT'][1]:
            state_confidence["Forward"] += 1
        elif self.ROLL_RANGES['FULL_DOWN'][0] <= roll <= self.ROLL_RANGES['FULL_DOWN'][1]:
            state_confidence["Looking Down"] += 1
            
        # Apply temporal smoothing
        current_state = max(state_confidence.items(), key=lambda x: x[1])[0]
        self.pose_history.append(current_state)
        if len(self.pose_history) > self.history_size:
            self.pose_history.pop(0)
            
        # Get most common state in history
        if len(self.pose_history) >= 3:
            from collections import Counter
            smoothed_state = Counter(self.pose_history).most_common(1)[0][0]
        else:
            smoothed_state = current_state
            
        # Debug information
        print(f"Debug - Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}")
        print(f"Confidence Scores: {state_confidence}")
        
        return smoothed_state

    def draw_pose_info(self, frame, angles, status):
        """Enhanced visualization with confidence scores"""
        if angles is not None:
            pitch, yaw, roll = angles
            
            # Draw angles with reference ranges
            y_offset = 30
            angle_info = [
                f"Pitch: {pitch:.1f} (Straight: 80-90, Down: 54-80)",
                f"Yaw: {yaw:.1f} (Straight: 115-125, Down: 80-105)",
                f"Roll: {roll:.1f} (Straight: -5-0, Down: -3-2)"
            ]
            
            for info in angle_info:
                cv2.putText(frame, info, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            # Draw status with color coding
            color = {
                "Forward": (0, 255, 0),
                "Partial Down": (0, 255, 255),
                "Looking Down": (0, 0, 255),
                "Looking Left": (255, 0, 0),
                "Looking Right": (255, 0, 0)
            }.get(status, (255, 255, 255))
            
            cv2.putText(frame, f"Status: {status}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        return frame 