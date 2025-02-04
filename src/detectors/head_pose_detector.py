import cv2 as cv
import numpy as np
import math

class HeadPoseDetector:
    def __init__(self):
        """Initialize head pose detector with 3D model points"""
        # 3D model points for face landmarks
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ])

        # Threshold angles for head pose detection
        self.angle_threshold = 30  # Degrees

    def get_pose(self, frame: np.ndarray, landmarks: np.ndarray):
        """
        Estimate head pose from facial landmarks
        
        Args:
            frame: Input image
            landmarks: Facial landmarks from YuNet (5 points)
            
        Returns:
            angles: (vertical_angle, horizontal_angle)
            status: Head pose status message
        """
        # Get image dimensions
        size = frame.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        
        # Camera matrix
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        # Convert YuNet landmarks to required format
        # YuNet provides: right eye, left eye, nose tip, right mouth, left mouth
        image_points = np.array([
            landmarks[2],  # Nose tip
            landmarks[2] + [0, 100],  # Chin (approximated)
            landmarks[1],  # Left eye
            landmarks[0],  # Right eye
            landmarks[4],  # Left mouth
            landmarks[3]   # Right mouth
        ], dtype="double")

        dist_coeffs = np.zeros((4,1))  # No lens distortion

        # Solve for pose
        success, rotation_vector, translation_vector = cv.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None, "Failed to estimate pose"

        # Convert rotation vector to angles
        rotation_matrix, _ = cv.Rodrigues(rotation_vector)
        pose_angles = self._rotation_matrix_to_angles(rotation_matrix)
        
        # Get pose status
        status = self._get_pose_status(pose_angles)
        
        return pose_angles, status

    def _rotation_matrix_to_angles(self, rotation_matrix):
        """Convert rotation matrix to angles"""
        x = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
        y = math.atan2(-rotation_matrix[2,0], math.sqrt(rotation_matrix[2,1]**2 + rotation_matrix[2,2]**2))
        z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
        
        # Convert to degrees
        x = math.degrees(x)
        y = math.degrees(y)
        z = math.degrees(z)
        
        return np.array([x, y, z])

    def _get_pose_status(self, angles):
        """Determine head pose status based on angles"""
        x, y, z = angles
        
        if abs(x) > self.angle_threshold:
            return "Head tilted" + (" right" if x > 0 else " left")
        elif abs(y) > self.angle_threshold:
            return "Head " + ("down" if y > 0 else "up")
        elif abs(z) > self.angle_threshold:
            return "Head turned" + (" right" if z > 0 else " left")
        
        return "Forward" 