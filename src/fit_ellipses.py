import numpy as np
import cv2


def fit_ellipses(contours: np.ndarray, num_eggs: int) -> np.ndarray:
    """
    Fit ellipse to contour
    :param num_eggs: number of eggs
    :param contours: Contours to fit ellipses to
    :return: Ellipse
    """
    ellipses = []
    for i in range(num_eggs):
        ellipse = cv2.fitEllipse(contours[i])
        ellipses.append(ellipse)

    return ellipses
