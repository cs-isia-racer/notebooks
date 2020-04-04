import sys
import os
import time
from pathlib import Path
import lib

import cv2

ROOT = Path(os.path.realpath(__file__)).parent.parent

def crawl(root, out):
    start = time.time()
    counter = 0

    for file in root.glob("**/*.jpg"):
        out_dir = out / [p for p in file.parts if p.startswith("out.")][0]
        out_dir.mkdir(parents=True, exist_ok=True)


        img = cv2.imread(str(file))
        result_path = str(out_dir / file.name)
        cv2.imwrite(result_path, lib.cluster_filter(img) * 255)
        counter += 1
        print(f"\rDone {counter} files in {time.time() - start} seconds", end='')

if __name__ == '__main__':
    crawl(ROOT / "dataset", ROOT / "dataset_clusterified")
