import numpy as np

import cv2

from image_load import load_images_labels

recognizer = cv2.face.LBPHFaceRecognizer_create()

images, labels = load_images_labels("images_labels.json", "training")
cv2.destroyAllWindows()

# Perform the training
recognizer.train(images, np.array(labels))
recognizer.save("recognizer.yml")
