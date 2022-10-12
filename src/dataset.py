import numpy as np
import cv2
from .bound_eggs import fit_rectangles


def clutch_dataset(contours: np.ndarray, label: str, num_eggs: int) -> np.ndarray:
    """
    Create classification dataset from contours
    :param num_eggs: number of eggs
    :param contours: Contours to fit ellipses to
    :param label: Label for dataset
    :return: Dataset
    """
    dataset = []
    rectangles = fit_rectangles(contours, num_eggs)
    resized = [np.resize(x, (64, 64)) for x in rectangles]
    for i in range(num_eggs):
        dataset.append([resized[i], label])

    return np.asarray(dataset)


def full_dataset(contours: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Create classification dataset from contours
    :param contours: Contours to fit ellipses to
    :param labels: Labels for dataset
    :return: Dataset
    """
    dataset = []
    for i in range(len(contours)):
        dataset.extend(clutch_dataset(contours[i], labels[i], len(contours[i])))
    return np.asarray(dataset)