from sklearn.cluster import DBSCAN
import numpy as np
import pickle
import os
import math
from PIL import Image

print("[INFO] Loading encodings data...")
data = pickle.loads(open('faces/faces.pickle', "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]
for d in data:
    d['cluster'] = -1

print("[INFO] Clustering the data...")
clt = DBSCAN(metric="euclidean", n_jobs=-1, eps=0.32)
clt.fit(encodings)

labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] Unique face Count: {}".format(numUniqueFaces))

for labelID in labelIDs:
    print("[INFO] Faces for face ID: {}".format(labelID))
    idxs = np.where(clt.labels_ == labelID)[0]
    idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)
    face_count = len(idxs)
    grid_size = math.ceil(math.sqrt(face_count))
    current_face_number = 0
    face_grid = Image.new(mode = "RGB", size = (grid_size * 96, grid_size * 96))
  
    for i in idxs:
        image = Image.open('photos/'+data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        face = image.crop((left, top, right, bottom))
        face = face.resize((96, 96))
        
        x = (current_face_number % grid_size) * 96
        y = (current_face_number // grid_size) * 96
        face_grid.paste(face, box=(x, y))
        data[i]["cluster"] = labelID
        current_face_number += 1

    print("[INFO] Writing grid of faces for", labelID)
    face_grid.save('faces/' + (str(labelID) if labelID != -1 else 'unknown') + '.jpg')

print("[INFO] Writing encodings to file...")
f = open('faces/cluster.pickle', "wb")
f.write(pickle.dumps(data))
f.close()
