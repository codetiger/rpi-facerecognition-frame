import numpy as np
import pickle
import face_recognition
import cv2
from PIL import Image
import os
from threading import *
import time 
from display import Display
import logging

logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s', level=logging.INFO)

class App():
    def __init__(self):
        self.display = Display()
        self.display.init_display()
        self.display.print_system_info()
        self.display.clear()
        
        self.pictures = []
        self.track_img_ndex = 0
        self.face_clusters_found = self.prev_face_clusters_found = []
        self.cam_image = None

    def start_threads(self):
        self.t1 = Thread(target=self.find_faces) 
        self.t2 = Thread(target=self.show_slides) 
        self.t1.start()
        logging.info("Started camera thread...")
        self.t2.start()
        logging.info("Started display thread...")
    
    def load_face_data(self):
        self.data = pickle.loads(open('faces/cluster.pickle', "rb").read())
        self.known_face_encodings = []
        self.known_clusters = []
        for face in self.data:
            if face['cluster'] != -1:
                self.known_face_encodings.append(face['encoding'])
                self.known_clusters.append(face['cluster'])

    def get_camera_image(self):
        logging.info("Loading camera image...")
        camera = cv2.VideoCapture(0)
        ret, img = camera.read()

        if ret == False:
            logging.info("Not able to capture frame from camera...")
            img = None
        else:
            img = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        camera.release()
        return img
    
    def get_image_features(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = face_recognition.face_locations(img_gray, model='hog')
        logging.info("Detected face count " + str(len(boxes)))
        clusters = []

        if len(boxes) > 0:
            encodings = face_recognition.face_encodings(image, boxes)
            for (top, right, bottom, left), face_encoding in zip(boxes, encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                cluster = -1
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    cluster = self.known_clusters[best_match_index]
                if cluster >= 0:
                    clusters.append(cluster)

        return clusters

    def sort_remove_duplicates(self, array, min_occurences):
        new_array = [ele for ele in array if array.count(ele) >= min_occurences]
        new_array = sorted(set(new_array), key = lambda ele: new_array.count(ele)) 
        return new_array

    def set_new_pictures(self, image_paths):
        logging.info("Setting new pictures " + str(len(image_paths)))
        self.pictures = image_paths
        self.track_img_ndex = 0

    def find_faces(self):
        while True:
            self.cam_image = self.get_camera_image()
            if self.cam_image is not None:
                self.face_clusters_found = self.get_image_features(self.cam_image)

                if self.face_clusters_found != self.prev_face_clusters_found:
                    self.prev_face_clusters_found = self.face_clusters_found
                    logging.info("Recognised new faces..." + str(self.face_clusters_found))                
                    
                    related_faces = ['photos/'+face['imagePath'] for face in self.data if face['cluster'] in self.face_clusters_found]
                    image_paths = self.sort_remove_duplicates(related_faces, len(self.face_clusters_found))
                    self.set_new_pictures(image_paths)
                else:
                    logging.info("No new faces found..." + str(self.face_clusters_found))

            time.sleep(10)
        
    def show_slides(self):
        while True:
            if self.track_img_ndex >= len(self.pictures):
                self.track_img_ndex = 0

            if len(self.pictures) > 0:
                x = self.pictures[self.track_img_ndex]
                self.track_img_ndex += 1
                image = Image.open(x)
                self.display.set_main_image(image)

            if self.cam_image is not None:
                img = Image.fromarray(self.cam_image)
                self.display.set_cam_image(img)
                
            self.display.update()

            time.sleep(3)


if __name__ == '__main__':
    app = App()
    app.load_face_data()
    app.start_threads()

