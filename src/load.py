from urllib.request import urlopen
import numpy as np
import cv2
from typing import List


def load_image(url: str) -> np.ndarray:
    """
    Load image from url
    :param url: url of image
    :return: image
    """
    image = urlopen(url)
    image = image.read()
    image = cv2.imdecode(np.asarray(bytearray(image), dtype="uint8"), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_images(urls: list) -> List[np.ndarray]:
    """
    Load images from urls
    :param urls: list of urls
    :return: list of images
    """
    return [load_image(url) for url in urls]

def load_images_from_url_csv(path: str) -> List[np.ndarray]:
    """
    Load images from csv file
    :param path: path to csv file
    :return: list of images
    :rtype: List[np.ndarray]
    """
    with open(path, 'r') as f:
        urls = [x.split(',')[1] for x in f.read().splitlines() if len(x) > 0]
    return load_images(urls)
