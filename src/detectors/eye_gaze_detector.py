import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

class EyeGazeDetector:
    def __init__(self):
        """Initialize MediaPipe Face Mesh for eye tracking"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define eye landmarks
        self.LEFT_IRIS = [474, 475, 476, 477]   # Left iris landmarks
        self.RIGHT_IRIS = [469, 470, 471, 472]  # Right iris landmarks
        
        # Complete eye landmarks
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Thresholds
        self.VERTICAL_RATIO_THRESHOLD = 0.2
        self.HORIZONTAL_RATIO_THRESHOLD = 0.35
        
        # Updated ranges based on collected data
        self.VERTICAL_RANGES = {
            'CENTER': (0.35, 0.41),    # Looking straight (0.38-0.41)
            'TOP': (0.38, 0.42),       # Looking top (0.40-0.42)
            'BOTTOM': (0.35, 0.44),    # Looking bottom (0.37-0.44)
            'SUSPICIOUS_UP': (0.25, 0.35),   # Suspiciously looking up
            'SUSPICIOUS_DOWN': (0.45, 0.65)  # Suspiciously looking down
        }
        
        self.HORIZONTAL_RANGES = {
            'CENTER': (0.49, 0.52),      # Center view (0.50-0.52)
            'LEFT': (0.50, 0.53),        # Left side (0.51-0.53)
            'RIGHT': (0.47, 0.50),       # Right side (0.48-0.50)
            'SUSPICIOUS_LEFT': (0.20, 0.45),  # Suspiciously far left
            'SUSPICIOUS_RIGHT': (0.55, 0.80)  # Suspiciously far right
        }
        
        # Position scores for confidence calculation
        self.POSITION_SCORES = {
            'CENTER': 3,
            'NORMAL': 1,
            'SUSPICIOUS': 2
        }
        
        # History for temporal smoothing
        self.history_size = 5
        self.gaze_history = []

    def detect_gaze(self, frame):
        """
        Detect eye gaze direction
        Returns: (vertical_ratio, horizontal_ratio), gaze_direction
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return None, "No Face Detected"
            
        frame_height, frame_width = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0]
        
        # Get eye and iris positions
        left_eye = self._get_eye_coordinates(landmarks, self.LEFT_EYE, frame_width, frame_height)
        right_eye = self._get_eye_coordinates(landmarks, self.RIGHT_EYE, frame_width, frame_height)
        left_iris = self._get_eye_coordinates(landmarks, self.LEFT_IRIS, frame_width, frame_height)
        right_iris = self._get_eye_coordinates(landmarks, self.RIGHT_IRIS, frame_width, frame_height)
        
        # Calculate relative positions
        left_ratio = self._calculate_iris_position(left_eye, left_iris)
        right_ratio = self._calculate_iris_position(right_eye, right_iris)
        
        # Average the ratios from both eyes
        vertical_ratio = (left_ratio[1] + right_ratio[1]) / 2
        horizontal_ratio = (left_ratio[0] + right_ratio[0]) / 2
        
        # Determine gaze direction
        gaze_direction = self._determine_gaze_direction(vertical_ratio, horizontal_ratio)
        
        # Apply temporal smoothing
        self.gaze_history.append(gaze_direction)
        if len(self.gaze_history) > self.history_size:
            self.gaze_history.pop(0)
        
        smoothed_direction = self._get_smoothed_direction()
        
        return (vertical_ratio, horizontal_ratio), smoothed_direction

    def _get_eye_coordinates(self, landmarks, indices, frame_width, frame_height):
        """Convert landmark indices to coordinates"""
        coordinates = []
        for idx in indices:
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            coordinates.append([x, y])
        return np.array(coordinates)

    def _calculate_iris_position(self, eye_points, iris_points):
        """Calculate iris position relative to eye corners"""
        eye_left = np.min(eye_points[:, 0])
        eye_right = np.max(eye_points[:, 0])
        eye_top = np.min(eye_points[:, 1])
        eye_bottom = np.max(eye_points[:, 1])
        
        iris_center = np.mean(iris_points, axis=0)
        
        # Calculate horizontal and vertical ratios
        horizontal_ratio = (iris_center[0] - eye_left) / (eye_right - eye_left)
        vertical_ratio = (iris_center[1] - eye_top) / (eye_bottom - eye_top)
        
        return [horizontal_ratio, vertical_ratio]

    def _determine_gaze_direction(self, vertical_ratio, horizontal_ratio):
        """Enhanced gaze direction detection with tighter ranges"""
        state_confidence = {
            "Looking Center": 0,
            "Looking Top": 0,
            "Looking Bottom": 0,
            "Looking Left": 0,
            "Looking Right": 0,
            "Suspicious Movement": 0
        }
        
        # Vertical check
        if self.VERTICAL_RANGES['CENTER'][0] <= vertical_ratio <= self.VERTICAL_RANGES['CENTER'][1]:
            state_confidence["Looking Center"] += self.POSITION_SCORES['CENTER']
        elif self.VERTICAL_RANGES['TOP'][0] <= vertical_ratio <= self.VERTICAL_RANGES['TOP'][1]:
            state_confidence["Looking Top"] += self.POSITION_SCORES['NORMAL']
        elif self.VERTICAL_RANGES['BOTTOM'][0] <= vertical_ratio <= self.VERTICAL_RANGES['BOTTOM'][1]:
            state_confidence["Looking Bottom"] += self.POSITION_SCORES['NORMAL']
        elif vertical_ratio < self.VERTICAL_RANGES['SUSPICIOUS_UP'][0] or vertical_ratio > self.VERTICAL_RANGES['SUSPICIOUS_DOWN'][0]:
            state_confidence["Suspicious Movement"] += self.POSITION_SCORES['SUSPICIOUS']
            
        # Horizontal check
        if self.HORIZONTAL_RANGES['CENTER'][0] <= horizontal_ratio <= self.HORIZONTAL_RANGES['CENTER'][1]:
            state_confidence["Looking Center"] += self.POSITION_SCORES['CENTER']
        elif self.HORIZONTAL_RANGES['LEFT'][0] <= horizontal_ratio <= self.HORIZONTAL_RANGES['LEFT'][1]:
            state_confidence["Looking Left"] += self.POSITION_SCORES['NORMAL']
        elif self.HORIZONTAL_RANGES['RIGHT'][0] <= horizontal_ratio <= self.HORIZONTAL_RANGES['RIGHT'][1]:
            state_confidence["Looking Right"] += self.POSITION_SCORES['NORMAL']
        elif horizontal_ratio < self.HORIZONTAL_RANGES['SUSPICIOUS_LEFT'][0] or horizontal_ratio > self.HORIZONTAL_RANGES['SUSPICIOUS_RIGHT'][0]:
            state_confidence["Suspicious Movement"] += self.POSITION_SCORES['SUSPICIOUS']
        
        # Debug print
        print(f"Eye Gaze Debug - Vertical: {vertical_ratio:.2f}, Horizontal: {horizontal_ratio:.2f}")
        print(f"Confidence Scores: {state_confidence}")
        
        return max(state_confidence.items(), key=lambda x: x[1])[0]

    def _get_smoothed_direction(self):
        """Apply temporal smoothing to gaze direction"""
        if len(self.gaze_history) >= 3:
            from collections import Counter
            return Counter(self.gaze_history).most_common(1)[0][0]
        return self.gaze_history[-1] if self.gaze_history else "Unknown"

    def draw_gaze_info(self, frame, eye_ratios, gaze_direction):
        """Draw eye gaze information separately from head pose"""
        if eye_ratios is not None:
            vertical_ratio, horizontal_ratio = eye_ratios
            
            # Draw eye gaze information below head pose info
            y_offset = 150  # Start below head pose info
            info_lines = [
                f"Eye Vertical: {vertical_ratio:.2f}",
                f"  Center: {self.VERTICAL_RANGES['CENTER'][0]:.2f}-{self.VERTICAL_RANGES['CENTER'][1]:.2f}",
                f"  Down: {self.VERTICAL_RANGES['SUSPICIOUS_DOWN'][0]:.2f}-{self.VERTICAL_RANGES['SUSPICIOUS_DOWN'][1]:.2f}",
                f"  Up: {self.VERTICAL_RANGES['SUSPICIOUS_UP'][0]:.2f}-{self.VERTICAL_RANGES['SUSPICIOUS_UP'][1]:.2f}",
                f"Eye Horizontal: {horizontal_ratio:.2f}",
                f"  Center: {self.HORIZONTAL_RANGES['CENTER'][0]:.2f}-{self.HORIZONTAL_RANGES['CENTER'][1]:.2f}",
                f"  Left: {self.HORIZONTAL_RANGES['LEFT'][0]:.2f}-{self.HORIZONTAL_RANGES['LEFT'][1]:.2f}",
                f"  Right: {self.HORIZONTAL_RANGES['RIGHT'][0]:.2f}-{self.HORIZONTAL_RANGES['RIGHT'][1]:.2f}"
            ]
            
            for line in info_lines:
                cv2.putText(frame, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            # Draw gaze direction with color coding
            color = (0, 255, 0) if gaze_direction == "Looking Center" else (0, 0, 255)
            cv2.putText(frame, f"Eye Status: {gaze_direction}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame 
    