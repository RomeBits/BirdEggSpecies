from urllib.request import urlopen
import numpy as np
import cv2


def load_image(url: str):
    """
    Load image from url
    :param url: url of image
    :return: image
    """
    image = urlopen(url)
    image = image.read()
    image = cv2.imdecode(np.asarray(bytearray(image), dtype="uint8"), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
