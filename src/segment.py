import numpy as np
import cv2

from typing import List, Tuple


def egg_contours(
        img: np.ndarray,
        dp_thresh: Tuple[int, int] = (8, 23),
        area_thresh: int = 500) -> List[np.ndarray]:
    """
    Find contours of egg in image
    :param img: Image to find contours in (RGB)
    :param dp_thresh: Threshold for number of curves (low, high)
    :param area_thresh: Threshold for area of contour
    :return: List of egg contours
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Blur image
    ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter for ovular contours
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if (len(approx) > dp_thresh[0]) & (len(approx) < dp_thresh[1]) & (area > area_thresh):
            contour_list.append(contour)

    return contour_list
