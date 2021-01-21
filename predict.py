import numpy as np
import pickle
import face_recognition
import cv2
from PIL import Image, ImageTk
import tkinter as tk
import os
from threading import *
import time 

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.w = 512
        self.h = 384
        self.is_window_closed = False
        self.geometry("%dx%d+0+0" % (self.w, self.h))
        self.pictures = []
        self.picture_display = tk.Label(self)
        self.picture_display.pack(expand=True, fill="both")

        self.track_img_ndex = 0
        self.face_clusters_found = self.prev_face_clusters_found = []
        self.cam_image = None
        self.load_face_data()

        self.t1 = Thread(target=self.find_faces) 
        self.t2 = Thread(target=self.show_slides) 
        self.t1.start()
        self.t2.start()
                
    def load_face_data(self):
        self.data = pickle.loads(open('faces/cluster.pickle', "rb").read())
        self.known_face_encodings = []
        self.known_clusters = []
        for face in self.data:
            if face['cluster'] != -1:
                self.known_face_encodings.append(face['encoding'])
                self.known_clusters.append(face['cluster'])

    def get_camera_image(self):
        print("[INFO] Loading camera image...")
        camera = cv2.VideoCapture(0)
        ret, img = camera.read()

        if ret == False:
            print("[Error] Not able to capture frame from camera...")
            img = None
        else:
            img = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        camera.release()
        return img
    
    def get_image_features(self, image):
        boxes = face_recognition.face_locations(image, model='hog')
        print("[INFO] Detected face count ", len(boxes))
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
        print("[INFO] Setting new pictures: ", image_paths)
        self.pictures = image_paths
        self.track_img_ndex = 0
        self.picture_display.config(image=None)
        self.picture_display.image = None

    def find_faces(self):
        while not self.is_window_closed:
            self.cam_image = self.get_camera_image()
            if self.cam_image is not None:
                self.face_clusters_found = self.get_image_features(self.cam_image)

                if self.face_clusters_found != self.prev_face_clusters_found:
                    self.prev_face_clusters_found = self.face_clusters_found
                    print("[INFO] Recognised new faces...", self.face_clusters_found)                
                    
                    related_faces = ['photos/'+face['imagePath'] for face in self.data if face['cluster'] in self.face_clusters_found]
                    image_paths = self.sort_remove_duplicates(related_faces, len(self.face_clusters_found))
                    self.set_new_pictures(image_paths)
                else:
                    print("[INFO] No new faces found...", self.face_clusters_found)

            time.sleep(10)
        
    def resize_image(self, image, max_width, max_height):
        height, width = image.shape[:2]
        scaling_factor = max_height / float(height)
        if max_width/float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        new_image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            
        return new_image

    def set_image(self, new_image):
        if new_image is not None:
            new_image = ImageTk.PhotoImage(Image.fromarray(new_image))
            
        self.picture_display.config(image=new_image)
        self.picture_display.image = new_image

    def show_slides(self):
        while not self.is_window_closed:
            if self.track_img_ndex >= len(self.pictures):
                self.track_img_ndex = 0

            if len(self.pictures) > 0:
                x = self.pictures[self.track_img_ndex]
                self.track_img_ndex += 1
                original_image = cv2.imread(x)
                resized = None

                if original_image is not None:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    if self.cam_image is not None:
                        small_cam_image = self.resize_image(self.cam_image, 150, 150)
                        original_image[10:10+small_cam_image.shape[0], 10:10+small_cam_image.shape[1]] = small_cam_image

                    resized = self.resize_image(original_image, self.w, self.h)
                    self.title(os.path.basename(x))
                else:
                    print("[INFO] Image not loaded from file...")
                    
                self.set_image(resized)

            time.sleep(2)

    def on_close(self):
        print("[INFO] Window closed...")
        self.is_window_closed = True
        self.destroy()
        

app = App()
app.protocol("WM_DELETE_WINDOW", app.on_close)
app.mainloop()
