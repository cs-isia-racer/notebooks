import math
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Estimator:
    def predict_one(self, image):
        raise NotImplemented

    def score(self, X, y):
        y_pred = [self.predict_one(x) for x in X]
        return (
            mean_squared_error(y, y_pred),
            mean_absolute_error(y, y_pred),
        )


def load_images(path, rgb=True):
    dataset_path = Path(path).expanduser()

    for out_path in sorted(list(dataset_path.iterdir())):
        for img_path in sorted(list(out_path.iterdir())):
            if img_path.name.endswith(".jpg"):
                _, nb, steering, throttle = img_path.name.replace(".jpg", "").split("_")
                nb = int(nb)
                id = f"{out_path.name}.{nb:08}"
                steering = float(steering)
                throttle = float(throttle)
                img = cv2.imread(str(img_path))
                if rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                yield (id, img, steering, throttle)


TEST_RATIO = 0.2


def load_dataset(path):
    X = []
    y = []

    for _, img, steer, _ in load_images(path):
        X.append(img)
        y.append(steer)

    lim = int(TEST_RATIO * len(X))

    return np.array(X[lim:]), np.array(y[lim:]), np.array(X[:lim]), np.array(y[:lim])


def detect_lines(
    img,
    hough_t=40,  # Minimum number of votes (intersection in Hough grid cell)
    min_line_length=20,  #  minimum number of pixels making up a line
    max_line_gap=10,  # maximum gap in pixels between connectable line segments
):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    return cv2.HoughLinesP(
        img, rho, theta, hough_t, np.array([]), min_line_length, max_line_gap
    )


def _line_angle(x1, y1, x2, y2, angle_limit=90):
    angle = None
    if y2 == y1:
        angle = math.pi / 2 if x2 > x1 else -math.pi / 2
    else:
        angle = math.atan((x2 - x1) / (y2 - y1))

    # Only keep realistic angles
    if abs(180 / math.pi * angle) > angle_limit:
        return None

    return angle


def angle_from_lines(lines):
    # Angle between -30 and 30 (-1 and 1)

    angles = []
    lengths = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = _line_angle(x1, y1, x2, y2)
            if angle:
                angles.append(angle)
                lengths.append(max(abs(x2 - x1), abs(y2 - y1)))

    if not angles:
        return 0

    angles = np.array(angles)
    lengths = np.array(lengths)

    weighted_mean_angle = np.sum(angles * lengths) / np.sum(lengths)

    a = -180 / math.pi * np.mean(weighted_mean_angle)

    return min(30, max(-30, a)) / 30


def remove_clusters(img, max_size):
    """
    remove_clusters removes the groups of pixels having the same values
    bigger than the given limit, this allows us to remove the noise outside
    the circuit after a kmeans

    it zeroes the pixels on the image border
    """
    clusters = {}

    height, width = img.shape[:2]

    counter = 1

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            pix = img[y][x]
            loc = (y, x)

            n1, n2, n3, n4 = (y+1, x), (y-1, x), (y, x-1), (y, x+1)

            color = 0
            color = color or (pix == img[n1] and clusters.get(n1))
            color = color or (pix == img[n2] and clusters.get(n2))
            color = color or (pix == img[n3] and clusters.get(n3))
            color = color or (pix == img[n4] and clusters.get(n4))

            if not color:
                color = counter
                counter += 1

            # Not necessary and gives a small speedup
            # clusters[loc] = color
            clusters[n1] = pix == img[n1] and color
            clusters[n2] = pix == img[n2] and color
            clusters[n3] = pix == img[n3] and color
            clusters[n4] = pix == img[n4] and color

    sizes = Counter(clusters.values())
    res = np.zeros(img.shape)

    for y in range(1, img.shape[0] -1):
        for x in range(1, img.shape[1] - 1):
            if sizes[clusters[(y, x)]] > max_size:
                res[y][x] = 0
            else:
                res[y][x] = img[y][x]

    return res
