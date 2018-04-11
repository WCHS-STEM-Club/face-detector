import cv2

cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("recognizer.yml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    predict_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(predict_image)
    cv2.imshow("Recognizing Face", predict_image)
    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        print(nbr_predicted)
    cv2.waitKey(1000)
