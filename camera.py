
import threading
import binascii
from time import sleep
from utils import base64_to_pil_image, pil_image_to_base64

import scipy
import time
from scipy.spatial import distance as dist
import imutils
from imutils import face_utils
import numpy as np
import PIL
from PIL import Image

import dlib
import cv2

class Camera(object):
    def __init__(self, makeup_artist):
        self.to_process = []
        self.to_output = []
        self.makeup_artist = makeup_artist
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./landmarks.dat")
        self.data = []

        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()

    def process_one(self):
        if not self.to_process:
            return

        # input is an ascii string. 
        input_str = self.to_process.pop(0)

        # convert it to a pil image
        input_img = base64_to_pil_image(input_str).convert('RGB')

        image_cv2 = np.array(input_img) 
        # Convert RGB to BGR 
        image_cv2 = image_cv2[:, :, ::-1].copy() 
        gray_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray_image, 1)
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(gray_image, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            self.data.append(ear)
            
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(image_cv2, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(image_cv2, [rightEyeHull], -1, (0, 255, 0), 1)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image_cv2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(image_cv2, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(image_cv2, "Ear {:4.4}".format(ear), (x - 20, y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the input_img
            for (x, y) in shape:

                cv2.circle(image_cv2, (x, y), 1, (0, 0, 255), -1)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(np.asarray(image_cv2))


        # output_str is a base64 string in ascii
        output_str = pil_image_to_base64(im_pil)

        # convert eh base64 string in ascii to base64 string in _bytes_
        self.to_output.append(binascii.a2b_base64(output_str))

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.01)

    def enqueue_input(self, input):
        self.to_process.append(input)

    def get_frame(self):
        while not self.to_output:
            sleep(0.05)
        return self.to_output.pop(0)

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        # return the eye aspect ratio
        return ear

