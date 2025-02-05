import cv2 as cv
from pathlib import Path

from detectors.face_detector import FaceDetector
from detectors.head_pose_detector import HeadPoseDetector
from utils.visualization import visualize_detections, draw_pose_info

def main():
    # Initialize detectors
    model_path = str(Path("models/face_detection_yunet_2023mar.onnx"))
    face_detector = FaceDetector(model_path)
    head_pose_detector = HeadPoseDetector()
    
    # Open webcam
    cap = cv.VideoCapture(0)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect faces
        detections = face_detector.detect(frame)
        
        # Visualize face detections
        output = visualize_detections(frame, detections)
        
        # Process each detected face
        for det in detections:
            # Extract landmarks
            landmarks = det[4:14].reshape((5,2))
            
            # Get head pose
            angles, status = head_pose_detector.get_pose(frame)
            
            if angles is not None:
                # Draw pose information
                output = head_pose_detector.draw_pose_info(output, angles, status)
        
        # Show output
        cv.imshow('Face and Pose Detection', output)
        
        # Exit on 'q' press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main() 