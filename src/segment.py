import numpy as np
import cv2

from typing import List, Tuple


def ruler_contour(img: np.ndarray):
    """
    Find contours of ruler in image
    :param img: Image to find contours in (RGB)
    :return: List of ruler contours
    """
    x, y = img.shape[0:2]
    crop = img[x - (x // 20) * 2:-1, 0:y]
    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # Getting Lines: https://stackoverflow.com/questions/45322630/how-to-detect-lines-in-opencv
    low_threshold = 50
    high_threshold = 100
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return crop, edges

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 5  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = x // 20  # minimum number of pixels making up a line
    max_line_gap = 1  # maximum gap in pixels between connectable line segments
    line_image = np.copy(crop) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    return crop, line_image


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
        #https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

        # https://byjus.com/maths/greens-theorem/#:~:text=Therefore%2C%20the%20line%20integral%20defined,%E2%88%AB%20c%20x%20d%20y
        area = cv2.contourArea(contour)
        if (len(approx) > dp_thresh[0]) & (len(approx) < dp_thresh[1]) & (area > area_thresh):
            contour_list.append(contour)

    return contour_list


def optimize_segmentation_hypers(
        imgs: List[np.ndarray],
        dp_thresh_lower_bound_range: Tuple[int, int] = (1,10),
        dp_thresh_upper_bound_range: Tuple[int, int] = (10, 30),
        area_thresh_range: Tuple[int, int] = (100, 500),
        max_contours: int = 5
) -> Tuple[Tuple[int, int], int, int]:
    """
    Find optimal segmentation hyperparameters
    :param imgs: List of images to optimize segmentation hyperparameters on
    :param dp_thresh_lower_bound_range: Range of lower bounds for number of curves
    :param dp_thresh_upper_bound_range: Range of upper bounds for number of curves
    :param area_thresh_range: Range of area thresholds
    :return: Optimal segmentation hyperparameters
    """
    # Find optimal segmentation hyperparameters
    best_dp_thresh = None
    best_area_thresh = None
    best_score = 0
    for dp_thresh_lower_bound in range(dp_thresh_lower_bound_range[0], dp_thresh_lower_bound_range[1]):
        for dp_thresh_upper_bound in range(dp_thresh_upper_bound_range[0], dp_thresh_upper_bound_range[1]):
            for area_thresh in range(area_thresh_range[0], area_thresh_range[1]):
                # Calculate score
                score = 0
                for img in imgs:
                    contours = egg_contours(img, (dp_thresh_lower_bound, dp_thresh_upper_bound), area_thresh)
                    if len(contours) > 0 and len(contours) <= max_contours:
                        score += 1

                # Update best score
                if score > best_score:
                    best_score = score
                    best_dp_thresh = (dp_thresh_lower_bound, dp_thresh_upper_bound)
                    best_area_thresh = area_thresh

    return best_dp_thresh, best_area_thresh, best_score
