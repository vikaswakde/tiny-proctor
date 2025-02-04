import cv2 as cv
import numpy as np

class FaceDetector:
    def __init__(
        self,
        model_path: str,
        input_size: list = [320, 320],
        conf_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000
    ):
        """Initialize YuNet face detector
        
        Args:
            model_path: Path to YuNet ONNX model
          
            conf_threshold: Confidence threshold
            nms_threshold: Non-maximum suppression threshold
            top_k: Maximum number of detections to keep
        """
        self.model = cv.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=tuple(input_size),
            score_threshold=conf_threshold, 
            nms_threshold=nms_threshold,
            top_k=top_k
        )
        
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect faces in frame
        
        Args:
            frame: Input image/frame
            
        Returns:
            Array of face detections with format:
            [[x,y,w,h,conf,landmarks...], ...]
        """
        # Update input size to match frame
        h, w = frame.shape[:2]
        self.model.setInputSize((w, h))
        
        # Detect faces
        faces = self.model.detect(frame)
        
        # Return empty array if no faces found
        return np.empty(shape=(0, 15)) if faces[1] is None else faces[1] 