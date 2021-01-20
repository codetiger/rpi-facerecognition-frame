import numpy as np
import pickle

data = pickle.loads(open('faces/cluster.pickle', "rb").read())
for face in data:
    print(face['imagePath'], face['cluster'])
