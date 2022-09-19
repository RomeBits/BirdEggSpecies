import numpy as np
import cv2

from typing import List


def egg_contours(img: np.ndarray, canny_thresh: int = 100, arc_thresh: int = 100) -> List[np.ndarray]:
    """
    Find contours of egg in image
    :param img: Image to find contours in (RGB)
    :param canny_thresh: Threshold for canny edge detection
    :param arc_thresh: Threshold for arc length of contours
    :return: List of egg contours
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Blur image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply canny edge detection
    edges = cv2.Canny(blur, canny_thresh, canny_thresh * 2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter for circular contours
    contours = [c for c in contours if cv2.arcLength(c, True) > arc_thresh]

    return contours
