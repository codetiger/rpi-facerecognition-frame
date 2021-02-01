import numpy as np
import pickle

data = pickle.loads(open('faces/faces.pickle', "rb").read())
for face in data:
    print(face['imagePath'], face['loc'])
