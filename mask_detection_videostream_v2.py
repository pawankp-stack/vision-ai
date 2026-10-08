# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:55:57 2020

mask_detection_videostream_v1.py : detect faces with or without mask in a live 
stream and mark them in the predicted video.

@author: Pawan Kumar Prasad pawan.prasad@ust-global.com>
"""


# import the required packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from argparse import ArgumentParser
from datetime import datetime
from time import sleep
import numpy as np
# import pafy
import cv2
import os



# construct the argument parser and parse the arguments
parser = ArgumentParser(description='Predict using a trained  AI model')

# List of supported CL arguments.
required_args = parser.add_argument_group('Required Arguments')

required_args.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
required_args.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
required_args.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
# args = vars(required_args.parse_args())
args = parser.parse_args()


def countdown(n):
    for i in range(n,0,-1):
        print("\n", end='')
        print("[INFO] Initiating the live video streaming & prediction sequence!! --> {}".format(i),end="", sep=" ", flush=True)
        sleep(1)
        print(chr(27) + "[2J" + "\033[F")

# loading our saved face detector model
print("[INFO] loading face detector model...")
prototxtPath = os.path.join(args.face, "deploy.prototxt")
weightsPath = os.path.join(args.face,
	"res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# loading the trained face mask detector model 
print("[INFO] loading face mask detector model...")
model = load_model(args.model)
print("")
print("[INFO] Required models are loaded and ready to process the live stream!!")
print("[INFO] Initiating the live video streaming & prediction sequence!!")

# calling the countdown function
countdown(3)

# uncomment the below lines and update the URL with the IP Cam's URL generated.
# to work with IP Cam


video_codec = cv2.VideoWriter_fourcc(*"XVID")
fps_w = 12
width = 600
height =600
file_name_p = ("Test_V2_PREDICTED.avi")
out = cv2.VideoWriter(file_name_p, video_codec, fps_w, (width, height))
"""
url = "https://www.youtube.com/watch?v=5ZquAQ_E71c"
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="mp4")
"""

import urllib.request
URL = "http://192.168.1.13:8080//shot.jpg"
cap =  cv2.VideoCapture(URL)


"""
# uncomment the below line for WebCam feed
# Initiating the camera object
"""
#cap = cv2.VideoCapture(0)

num = 1

while (cap.isOpened()):
    # fps = cap.get(cv2.CAP_PROP_FPS)
    fps = cap.set(cv2.CAP_PROP_FPS, 10)
   
    # Reading frames from the live feed
    ret, image = cap.read()

    # uncomment the below two lines for IP Cam feed
    img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()),dtype=np.uint8)
    image = cv2.imdecode(img_arr,-1)

    # creating a copy of the frame for futher scaling and use
    orig_copy = image.copy()
    (height, width) = image.shape[:2]
    
    # creating a blob from the input image
    blob = cv2.dnn.blobFromImage(image, 1.0, (1080, 1080),
    	(104.0, 177.0, 123.0))
    
    # passing the blob through the face_detection model and get the faces in the image
    net.setInput(blob)
    detections = net.forward()
    
    # the detected faces are then run in a loop
    for i in range(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with
    	# the detection
    	confidence = detections[0, 0, i, 2]
    
        # check for the confidence, and clear out the once with confidence less
        # less than the minimum confidence
    	if confidence > args.confidence:
    		# obtain the coordinates of the bounding box for
    		# the detected faces
    		print("[INFO] faces detected...", fps)
    		box_cor = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
    		(sp_X, sp_Y, end_X, end_Y) = box_cor.astype("int")
    
    		# check if the bounding boxes are within the frame
    		(sp_X, sp_Y) = (max(0, sp_X), max(0, sp_Y))
    		(end_X, end_Y) = (min(width - 1, end_X), min(height - 1, end_Y))
            
            
            # extract the face ROI
    		face = image[sp_Y:end_Y, sp_X:end_X]
            
            # convert the extract from BGR to RGB channel
    		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            # resize it to 224x224
    		face = cv2.resize(face, (224, 224))
    		face = img_to_array(face)
            
            # preprocess the extract before feeding for predictions
    		face = preprocess_input(face)
    		face = np.expand_dims(face, axis=0)
            
            # passing the extracted face to the model for prediction
            # of masked or non-masked
    		(mask, withoutMask) = model.predict(face)[0]
            
            # choosing the class and color for writing and 
            # drawing on the frame
    		label = "Mask" if mask > withoutMask else "No Mask"
    		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    
    		# print the confidence % with the label
    		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
    		print("Labels", label)
            
            # drwaing and writing the boxes and labels on the frame
    		cv2.putText(image, label, (sp_X, sp_Y - 10),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    		cv2.rectangle(image, (sp_X, sp_Y), (end_X, end_Y), color, 2)
            # drwaing and writing the boxes and labels on the frame
    		# cv2.putText(image,str(datetime.now()),(140,250), cv2.FONT_HERSHEY_SIMPLEX, .5,(255,255,255),2,cv2.LINE_AA)
            

    # Streaming the predicted video
    image = cv2.resize(image, (600,600))
    cv2.imshow("Output", image)
    out.write(image)
    
    num += 1
    if cv2.waitKey(1) & 0xff==ord('q'):
        break

# release the video capture object and destroy the opened windows
cap.release()
cv2.destroyAllWindows()
