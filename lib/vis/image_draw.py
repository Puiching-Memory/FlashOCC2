import cv2
import numpy as np

def draw_3d22d_box(
    img: np.ndarray,
    box: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
):
    img = cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)
    img = cv2.line(img, (int(box[2]), int(box[3])), (int(box[4]), int(box[5])), color, thickness)
    img = cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, thickness)
      
    return img 
