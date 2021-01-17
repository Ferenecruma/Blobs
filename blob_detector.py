from glob import glob
from typing import List, Dict, Tuple
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
import matplotlib.patches as patches
from skimage.feature import blob_dog


def detect_blobs(folder_path: str) -> Tuple[List[Dict], Image.Image]:
    result = []
    image_paths = glob(folder_path + "/*.png")

    images = [Image.open(path) for path in image_paths]

    # Convert image to grayscale, invert and convert to numpy array.
    np_images = map(
        lambda image: np.asarray(ImageOps.invert(image.convert("L")), float), images
    )

    for path, image in zip(image_paths, np_images):
        blobs = blob_dog(image, min_sigma=0.7, max_sigma=2, threshold=10)
        blobs[:, 2] = blobs[:, 2] * sqrt(2)

        radius = blobs[0][2]

        y_min, y_max = np.min(blobs[:, 0]) - radius, np.max(blobs[:, 0]) + radius
        x_min, x_max = np.min(blobs[:, 1]) - radius, np.max(blobs[:, 1]) + radius

        result.append({"file": path, "coords": [x_min, x_max, y_min, y_max]})

    for image, item in zip(images, result):
        x_min, x_max, y_min, y_max = item["coords"]

        draw = ImageDraw.Draw(image)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red")

    return result, images


res, images = detect_blobs("/home/roman/Projects/it_jim_image_processing/Blobs")
print(res)
for idx, image in enumerate(images):
    image.save(f"./detected/{idx}.jpg")
