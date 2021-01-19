import numpy as np
import pickle
import face_recognition
import cv2

data = pickle.loads(open('faces/cluster.pickle', "rb").read())
known_face_encodings = []
known_labels = []
for face in data:
    known_face_encodings.append(face['encoding'])
    known_labels.append(face['cluster'])

cluster_names = []
face_count = 0
print("[INFO] Loading test image...")
img = cv2.imread('test.jpeg')
if img is not None:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(img, model='hog')
    face_count = len(boxes)
    print("[INFO] Detected face count ", face_count)

    encodings = face_recognition.face_encodings(img, boxes)
    for (top, right, bottom, left), face_encoding in zip(boxes, encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        cluster = -1
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            cluster = known_labels[best_match_index]
            
        cluster_names.append(cluster)
        
related_faces = []
for face in data:
    related_faces = [face['imagePath'] for face in data if face['cluster'] in cluster_names]

images = []
for face in related_faces:
    if related_faces.count(face) >= face_count:
        images.append(face)
        
images = sorted(set(images), key = lambda ele: images.count(ele)) 
print(images)