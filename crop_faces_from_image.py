# how to execute
# ex) if you want to make face numbering start from 10 (default: 1), execute as follows:
# python crop_faces_from_image.py -n 10

# import numpy
import numpy as np

# module for system control
import os
import argparse

# module for webcam and capturing a webcam screenshot
import cv2
from PIL import Image

# modules for face recognition
import face_recognition
import pickle

# construct the argument parser and parse the arguments
INPUT = 'input' # directory containing images that have faces
OUTPUT = 'output' # directory of cropped faces
MODEL = 'face_detector'
CONFIDENCE = 0.5

# load the serialized face detector model
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([MODEL, "deploy.prototxt"])
weightsPath = os.path.sep.join([MODEL, "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# set the starting number of the output faces
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--number', type = int, default = 1, help = 'starting number of the output faces')
n = vars(ap.parse_args())['number']

for file in os.listdir(INPUT):
    # load the input image and grab the image spatial dimensions
    image = cv2.imread(f'{INPUT}/{file}')
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > CONFIDENCE:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int") # pixels of image

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = Image.open(f'{INPUT}/{file}').crop((startX, startY, endX, endY))
            face.save(f'{OUTPUT}/face{n}.jpg')
    
    # add 1 to n
    n += 1