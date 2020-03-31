from pathlib import Path

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

    for out_path in dataset_path.iterdir():
        for img_path in out_path.iterdir():
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
