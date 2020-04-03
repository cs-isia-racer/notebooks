import math
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from sklearn.cluster import KMeans

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


def load_dataset(path, rgb=True):
    X = []
    y = []

    for _, img, steer, _ in load_images(path, rgb=rgb):
        X.append(img)
        y.append(steer)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=42)
    return X_train, y_train, X_test, y_test


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


ANGLE_LIMIT = 35


def angle_from_lines(lines, weighted):
    # Angle between -ANGLE_LIMIT and ANGLE_LIMIT (-1 and 1)

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

    if not weighted:
        lengths = np.ones(lengths.shape)

    weighted_mean_angle = np.sum(angles * lengths) / np.sum(lengths)

    a = -180 / math.pi * np.mean(weighted_mean_angle)

    return min(ANGLE_LIMIT, max(-ANGLE_LIMIT, a)) / ANGLE_LIMIT


def draw_lines(img, lines, offset=0, weighted=False):
    line_image = np.zeros_like(img)

    for line in lines:
        for x1, y1, x2, y2 in line:
            color = 42
            cv2.line(line_image, (x1, y1 + offset), (x2, y2 + offset), color, 10)

    dy = 30
    dx = int(dy * math.tan(angle_from_lines(lines, weighted)))
    x = img.shape[1] // 2
    y = img.shape[0] - 1

    cv2.arrowedLine(line_image, (x, y), (x + dx, y - dy), 7, 5)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    return lines_edges


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

    seen = set()
    for yy in range(height):
        for xx in range(width):
            if (yy, xx) in seen:
                continue

            stack = [(yy, xx)]
            while stack:
                loc = stack.pop()
                if loc in seen:
                    continue

                seen.add(loc)
                (y, x) = loc

                # Boundaries
                if x * y == 0 or x >= width - 1 or y >= height - 1:
                    continue

                pix = img[y][x]

                neighbors = (
                    n
                    for n in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1))
                    if pix == img[n]
                )

                color = counter
                for n in neighbors:
                    stack.append(n)
                    color = clusters.get(n, color)

                counter += 1

                clusters[loc] = color
                for n in neighbors:
                    clusters[n] = color

    sizes = Counter(clusters.values())
    res = np.zeros(img.shape)

    for y in range(1, img.shape[0] - 1):
        for x in range(1, img.shape[1] - 1):
            if sizes[clusters[(y, x)]] > max_size:
                res[y][x] = 0
            else:
                res[y][x] = img[y][x]

    return res


# We train a KMeans to identify two groups of pixels:
# - The "black pixels"
# - The "white pixels"
# This is much more efficient than using a binary filter
def find_cluster_centers(X):
    length, h, w = X.shape[:3]
    kmeans = KMeansAlgo(
        n_clusters=2,
        random_state=0,
        verbose=True,
        compute_labels=False,
        batch_size=2000,
        n_init=10,
    ).fit(X.reshape(length * h * w, 3))
    return kmeans


# Obtained using kmeans
DEFAULT_CLUSTER_CENTERS = np.array(
    [
        [190.92414031, 143.03555383, 142.36439433],
        [74.92775525, 49.96258754, 43.38757833],
    ]
)


def l2_dist(u, v):
    d = u - v
    d2 = (d * d).sum(axis=2)
    return d2


def cluster_filter(img, centers=DEFAULT_CLUSTER_CENTERS):
    c1, c2 = centers

    img = img.astype(np.float64)
    d1 = l2_dist(img, c1)
    d2 = l2_dist(img, c2)

    return (d1 > d2).astype(np.uint8)
