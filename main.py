import cv2 as cv
import pygame

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

cap = cv.VideoCapture(0)

pygame.mixer.init()
pygame.mixer.music.load("speech.wav")
pygame.mixer.music.play()

last_face_countdown = 0
prev_last_face_countdown = 0
max_countdown = 50

while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0 and last_face_countdown > 1:
        last_face_countdown = 0
    elif len(faces) > 0 and last_face_countdown <= 1:
        last_face_countdown = 1
    else:
        last_face_countdown = min(last_face_countdown + 1, max_countdown)
    print(last_face_countdown)

    if last_face_countdown == 0 and prev_last_face_countdown == max_countdown:
        pygame.mixer.music.play()
        print("Test!")

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv.imshow('img', img)
    prev_last_face_countdown = last_face_countdown
    k = cv.waitKey(10) & 0xff
    if k == 27:
        break
cv.destroyAllWindows()
