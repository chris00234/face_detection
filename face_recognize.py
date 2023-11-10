import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = ['chris', 'jenna']

# features = np.load('features.npy')
# lables = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml') #read saved train data

img = cv.imread(r'/Users/chrischo/Documents/Computer_Vision_AI/facedetection/people/val/jenna/IMG_2350.JPG')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

face_rect = haar_cascade.detectMultiScale(gray, 1.1, 4) #detect face
for (x,y,w,h) in face_rect:
    face_roi = gray[y:y+h,x:x+w]
    
    label, confidence = face_recognizer.predict(face_roi)
    print(f'label = {people[label]} with confidence of {confidence}')
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255, 0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness= 2)

cv.imshow('Detected Face', img)

cv.waitKey(0)