import os
import cv2 as cv
import numpy as np

p = []
DIR = r'/Users/chrischo/Documents/Computer_Vision_AI/facedetection/people/train'
for i in os.listdir(DIR):
    p.append(i)

haar_cascade = cv.CascadeClassifier('haar_face.xml')
features = []
labels = []
def create_train():
    for person in p:
        path = os.path.join(DIR, person) # go to specific person dir
        label = p.index(person)
        print(label)
        
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            
            for(x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
                
create_train()

# print(f'length of the features = {len(features)}')
# print(f'length of the labels = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

#train the recognizer on the features list and the labels list
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)