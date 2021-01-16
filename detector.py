import os
import cv2
import glob
import pickle
import numpy as np
import face_recognition

if not os.path.exists('faces'):
    os.makedirs('faces')
data = []
files = glob.glob('photos/*.jpg') + glob.glob('photos/*.jpeg')

for i, filepath in enumerate(files):
    folder = os.path.split(filepath)[0]
    filename = os.path.split(filepath)[1].split(".")[0]
    fileext = os.path.split(filepath)[1].split(".")[1]

    print("[INFO] Analysing file ", filename + '.' + fileext, " ", str(i), "/", str(len(files)))
    img = cv2.imread(filepath)

    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(img, model='hog')
        print("[INFO] Detected face count ", len(boxes))
    
        encodings = face_recognition.face_encodings(img, boxes)
        d = [{"imagePath": filename+'.'+fileext, "loc": box, "encoding": enc}
            for (box, enc) in zip(boxes, encodings)]
        data.extend(d)
        print("[INFO] Encoding face data...")

print("[INFO] Writing encodings to file...")
f = open('faces/faces.pickle', "wb")
f.write(pickle.dumps(data))
f.close()
