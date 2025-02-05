import cv2 as cv
from pathlib import Path

from detectors.face_detector import FaceDetector
from detectors.head_pose_detector import HeadPoseDetector
from detectors.eye_gaze_detector import EyeGazeDetector
from utils.visualization import visualize_detections

def main():
    # Initialize detectors
    model_path = str(Path("models/face_detection_yunet_2023mar.onnx"))
    face_detector = FaceDetector(model_path)
    head_pose_detector = HeadPoseDetector()
    eye_gaze_detector = EyeGazeDetector()
    
    # Open webcam
    cap = cv.VideoCapture(0)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Initialize status variables
        head_status = "Forward"  # Default status
        angles = None
            
        # Detect faces
        detections = face_detector.detect(frame)
        output = visualize_detections(frame, detections)
        
        # Process head pose
        angles, head_status = head_pose_detector.get_pose(frame)
        if angles is not None:
            output = head_pose_detector.draw_pose_info(output, angles, head_status)
            
        # Process eye gaze
        eye_ratios, gaze_direction = eye_gaze_detector.detect_gaze(frame)
        if eye_ratios is not None:
            output = eye_gaze_detector.draw_gaze_info(output, eye_ratios, gaze_direction)
        
        # Combined analysis
        is_suspicious = (
            head_status in ["Looking Down", "Looking Left", "Looking Right"] or
            gaze_direction in ["Looking Down", "Looking Left", "Looking Right"]
        )
        
        # Draw warning if suspicious
        if is_suspicious:
            y_offset = 250  # Below both head and eye info
            cv.putText(output, "WARNING: Suspicious Activity", (10, y_offset),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show output
        cv.imshow('Proctoring', output)
        
        # Exit on 'q' press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main() 