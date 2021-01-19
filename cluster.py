from sklearn.cluster import DBSCAN
import numpy as np
import pickle
import cv2
import os

def move_image(image,id,labelID):	
	path = 'faces/label'+str(labelID)
	if os.path.exists(path) == False:
		os.mkdir(path)

	filename = str(id) +'.jpg'	
	cv2.imwrite(os.path.join(path , filename), image)
	return


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
print("[INFO] # unique faces: {}".format(numUniqueFaces))

for labelID in labelIDs:
    print("[INFO] Faces for face ID: {}".format(labelID))
    idxs = np.where(clt.labels_ == labelID)[0]
    idxs = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)

    for i in idxs:
        image = cv2.imread('photos/'+data[i]["imagePath"])
        (top, right, bottom, left) = data[i]["loc"]
        face = image[top:bottom, left:right]
        face = cv2.resize(face, (96, 96))
        move_image(face,i,labelID)
        data[i]["cluster"] = labelID

print("[INFO] Writing encodings to file...")
f = open('faces/cluster.pickle', "wb")
f.write(pickle.dumps(data))
f.close()
