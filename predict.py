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
        self.geometry("%dx%d+0+0" % (self.w, self.h))
        self.pictures = []
        self.track_img_ndex = 0
        self.picture_display = tk.Label(self)
        self.picture_display.pack(expand=True, fill="both")
        self.is_window_closed = False
        self.t1 = Thread(target=self.find_faces) 
        self.t1.start()
        self.t2 = Thread(target=self.show_slides) 
        self.t2.start()
        
        self.data = pickle.loads(open('faces/cluster.pickle', "rb").read())
        self.known_face_encodings = []
        self.known_labels = []
        for face in self.data:
            self.known_face_encodings.append(face['encoding'])
            self.known_labels.append(face['cluster'])
        self.face_clusters_found = []
        self.prev_face_clusters_found = []
        self.cam_image = None

    def find_faces(self):
        while not self.is_window_closed:
            self.face_clusters_found = []
            face_count = 0
            print("[INFO] Loading camera image...")
            camera = cv2.VideoCapture(0)
            ret, img = camera.read()
            
            if ret == False:
                print("[Error] Not able to capture frame from camera...")
                self.cam_image = None
                img = None
            else:
                self.cam_image = img.copy()
                self.cam_image = cv2.cvtColor(self.cam_image, cv2.COLOR_BGR2RGB)
                
                if self.cam_image is not None:
                    boxes = face_recognition.face_locations(self.cam_image, model='hog')
                    face_count = len(boxes)
                    print("[INFO] Detected face count ", face_count)

                    if face_count > 0:
                        encodings = face_recognition.face_encodings(self.cam_image, boxes)
                        for (top, right, bottom, left), face_encoding in zip(boxes, encodings):
                            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                            cluster = -1
                            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                cluster = self.known_labels[best_match_index]
                            
                            if cluster != -1:
                                self.face_clusters_found.append(cluster)

                        if self.face_clusters_found != self.prev_face_clusters_found:
                            print("[INFO] Found new faces...", self.face_clusters_found)
                        
                            self.prev_face_clusters_found = self.face_clusters_found
                            related_faces = ['photos/'+face['imagePath'] for face in self.data if face['cluster'] in self.face_clusters_found and face['cluster'] != -1]
                            print("Found new pictures: ", related_faces)

                            images = []
                            for face in related_faces:
                                if related_faces.count(face) >= len(self.face_clusters_found):
                                    images.append(face)
                                    
                            images = sorted(set(images), key = lambda ele: images.count(ele)) 
                            self.pictures = images
                            print("Setting new pictures: ", self.pictures)
                            self.track_img_ndex = 0
                            self.picture_display.config(image=None)
                            self.picture_display.image = None
                        else:
                            print("[INFO] No new faces found...", self.face_clusters_found)

            camera.release()
            time.sleep(10)
        
    def resize_image(self, image, max_width, max_height):
        height, width = image.shape[:2]
        scaling_factor = max_height / float(height)
        if max_width/float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        new_image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            
        return new_image

    def show_slides(self):
        while not self.is_window_closed:
            if self.track_img_ndex >= len(self.pictures):
                self.track_img_ndex = 0

            if len(self.pictures) > 0:
                x = self.pictures[self.track_img_ndex]
                self.track_img_ndex += 1
                original_image = cv2.imread(x)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                if original_image is not None:
                    if self.cam_image is not None:
                        small_cam_image = self.resize_image(self.cam_image, 150, 150)
                        original_image[10:10+small_cam_image.shape[0], 10:10+small_cam_image.shape[1]] = small_cam_image

                    resized = self.resize_image(original_image, self.w, self.h)
                    if resized is not None:
                        new_img = ImageTk.PhotoImage(Image.fromarray(resized))
                        self.picture_display.config(image=new_img)
                        self.picture_display.image = new_img
                    else:
                        self.picture_display.config(image=None)
                        self.picture_display.image = None
                        print("[INFO] Image resizing failed...")
                    self.title(os.path.basename(x))
                else:
                    self.picture_display.config(image=None)
                    self.picture_display.image = None
                    print("[INFO] Image not loaded from file...")

            time.sleep(2)

    def on_close(self):
        print("[INFO] Window closed...")
        self.is_window_closed = True
        self.destroy()
        

app = App()
app.protocol("WM_DELETE_WINDOW", app.on_close)
app.mainloop()
