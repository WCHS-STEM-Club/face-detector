import cv2

from image_load import load_images_labels

cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.face.LBPHFaceRecognizer_create()

testing_images, testing_labels = load_images_labels("images_labels.json", "testing")
cv2.destroyAllWindows()

recognizer.read("recognizer.yml")

for i in range(len(testing_images)):
    predict_image = testing_images[i]
    nbr_actual = testing_labels[i]
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        if nbr_actual == nbr_predicted:
            print("{} is Correctly Recognized with confidence {}".format(nbr_actual, conf))
        else:
            print("{} is Incorrectly Recognized as {}".format(nbr_actual, nbr_predicted))
        cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
        cv2.waitKey(1000)
