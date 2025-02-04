import cv2 as cv
import numpy as np

def visualize_detections(
    image: np.ndarray,
    detections: np.ndarray,
    box_color: tuple = (0, 255, 0),
    text_color: tuple = (0, 0, 255),
    landmark_colors: list = None
) -> np.ndarray:
    """Draw face detections on image
    
    Args:
        image: Input image
        detections: Array of face detections
        box_color: Color for bounding box
        text_color: Color for confidence text
        landmark_colors: Colors for facial landmarks
        
    Returns:
        Image with visualizations
    """
    output = image.copy()
    
    if landmark_colors is None:
        landmark_colors = [
            (255, 0, 0),    # right eye
            (0, 0, 255),    # left eye
            (0, 255, 0),    # nose tip
            (255, 0, 255),  # right mouth corner
            (0, 255, 255)   # left mouth corner
        ]

    # Draw each detection
    for det in detections:
        # Draw bounding box
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(
            output,
            (bbox[0], bbox[1]),
            (bbox[0]+bbox[2], bbox[1]+bbox[3]),
            box_color,
            2
        )

        # Draw confidence score
        conf = det[-1]
        cv.putText(
            output,
            f'{conf:.2f}',
            (bbox[0], bbox[1]-10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color
        )

        # Draw landmarks
        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(
                output,
                landmark,
                2,
                landmark_colors[idx],
                2
            )

    return output

def draw_pose_info(
    image: np.ndarray,
    angles: np.ndarray,
    status: str,
    position: tuple = (20, 50),
    color: tuple = (0, 255, 0)
) -> np.ndarray:
    """Draw head pose information on image
    
    Args:
        image: Input image
        angles: (x,y,z) angles from pose estimation
        status: Pose status message
        position: Position to draw text
        color: Text color
        
    Returns:
        Image with pose information
    """
    output = image.copy()
    
    # Draw pose status
    cv.putText(
        output,
        f"Pose: {status}",
        position,
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )
    
    # Draw angles
    cv.putText(
        output,
        f"X: {angles[0]:.1f}, Y: {angles[1]:.1f}, Z: {angles[2]:.1f}",
        (position[0], position[1] + 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )
    
    return output 