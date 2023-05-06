import numpy as np
import argparse
import imutils
import time
import cv2
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from imutils.video import VideoStream
import tensorflow.keras.backend as K
import dlib


# Load CNN face detector into dlib
dnnFaceDetector=dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")  # <== put here the path of dlib detector

# Load saved model
print("[MSg] loading classifier model...")
classifier_model=tf.keras.models.load_model('face_classifier_model.h5') #<== put here the path of your classifier model
print("[MSg] loading vgg_face model...")
vgg_face=tf.keras.models.load_model('vgg_face_model.h5') #<== put here the path of your vgg_face model

# this line MUST be taken from the  variable "person_rep" printed after training-- find it in the nootbook of face_recognition project 
person_rep = {0: 'Jackman',
                1: 'Merkel',
                2: 'Hawking',
                3: 'Jolie',
                4: 'Macron',
                5: 'Olusegun',
                6: 'Washington'}



# initialize the video stream and allow the camera sensor to warm up
print("[MSg] starting video stream...")

vs = VideoStream(src=0).start()

time.sleep(2.0)

# Define the fps to be equal to 10. Also frame size is passed.

#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 26, (640,480))
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	#succs,frame = vs.read()
	frame = vs.read()
	#print(frame.shape)
	#frame = cv2.resize(frame, (300, 400))
	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Detect Faces
	# Detect Faces
	rects=dnnFaceDetector(gray,1)
	left,top,right,bottom=0,0,0,0
	
	for (i,rect) in enumerate(rects):
		# Extract Each Face
		left=rect.rect.left() #x1
		top=rect.rect.top() #y1
		right=rect.rect.right() #x2
		bottom=rect.rect.bottom() #y2
		width=right-left
		height=bottom-top
		img_crop=frame[top:top+height,left:left+width]
		img_crop = cv2.resize(img_crop, (224, 224))
		img_crop=img_to_array(img_crop)
		img_crop=np.expand_dims(img_crop,axis=0)
		img_crop=preprocess_input(img_crop)
		img_encode=vgg_face(img_crop)
		# Make Predictions
		embed=K.eval(img_encode)
		person=classifier_model.predict(embed)
		name=person_rep[np.argmax(person)]

		if np.max(person) < 0.9:
			name = "Unknown"
		
        confidence = "{:.2%}".format(np.max(person))

	    cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0), 2)
		frame=cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),3,cv2.LINE_AA)
		frame=cv2.putText(frame, confidence, (right,bottom+10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1,cv2.LINE_AA)


