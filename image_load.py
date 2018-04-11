import cv2
import json

import numpy as np
from PIL import Image


cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


def load_images_labels(json_file, set):
    images = []
    labels = []

    with open(json_file) as images_labels_file:
        images_labels = json.load(images_labels_file)
        for label in range(len(images_labels)):
            set_image_paths = images_labels[label][set]
            for image_path in set_image_paths:
                image_pil = Image.open(image_path).convert("L")
                image = np.array(image_pil, "uint8")
                faces = faceCascade.detectMultiScale(image)
                for (x, y, w, h) in faces:
                    images.append(image[y: y + h, x: x + w])
                    labels.append(label)

    return images, labels
