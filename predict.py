import numpy as np
import pickle
import face_recognition
import cv2
from PIL import Image
import os
from threading import *
import time 
from IT8951 import constants

class Display():
    display = None
    
    def init_display(self, virtual=False):
        print('[INFO] Initializing E-Paper Display')
        # Rotation: 'None', 'CW', 'CCW', 'flip'
        if virtual:
            from IT8951.display import VirtualEPDDisplay
            self.display = VirtualEPDDisplay(dims=(1872, 1404), rotate=None)
        else:
            from IT8951.display import AutoEPDDisplay
            self.display = AutoEPDDisplay(vcom=-2.06, rotate=None, spi_hz=24000000)
            print('[INFO] VCOM set to', self.display.epd.get_vcom())
        
    def print_system_info(self):
        epd = self.display.epd
        print('[INFO] Display info:')
        print('  display size: {}x{}'.format(self.display.epd.width, self.display.epd.height))
        print('  img buffer address: {:X}'.format(self.display.epd.img_buf_address))
        print('  firmware version: {}'.format(self.display.epd.firmware_version))
        print('  LUT version: {}'.format(self.display.epd.lut_version))

    def clear(self):
        print('[INFO] Clearing display...')
        self.display.clear()

    def display_image(self, image):
        self.display.frame_buf.paste(0xFF, box=(0, 0, self.display.width, self.display.height))
        dims = (self.display.width, self.display.height)
        image.thumbnail(dims)
        paste_coords = [dims[i] - image.size[i] for i in (0,1)]  # align image with bottom of display
        self.display.frame_buf.paste(image, paste_coords)
        self.display.draw_full(constants.DisplayModes.GC16)

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
        print("[INFO] Started camera thread...")
        self.t2.start()
        print("[INFO] Started display thread...")
    
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
            print("[INFO] Camera image shape", img.shape)
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
        self.display.clear()

    def find_faces(self):
        while True:
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
            new_image = Image.fromarray(new_image)
            self.display.display_image(new_image)

    def show_slides(self):
        while True:
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

                    resized = self.resize_image(original_image, self.display.width, self.display.height)
                else:
                    print("[INFO] Image not loaded from file...")
                    
                self.set_image(resized)
            else:
                self.set_image(self.cam_image)

            time.sleep(3)


if __name__ == '__main__':
    app = App()
    app.load_face_data()
    app.start_threads()

