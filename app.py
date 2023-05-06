import cv2
import streamlit as st
import numpy as np
import helper

# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture(0)

# Set the resolution of the camera
cap.set(3, 640)
cap.set(4, 480)

def get_frame():
    ret, frame = cap.read()
    return frame

# Display the live video stream using Streamlit
st.title('Live Camera Stream')
run = st.checkbox('Run')
if run:
    stframe = st.empty()
    while True:
        frame = get_frame()
        predict_frame = helper.predict_frame(frame)

        # Convert the frame from OpenCV's BGR format to RGB format for display in Streamlit
        frame = cv2.cvtColor(predict_frame, cv2.COLOR_BGR2RGB)

        # Display the frame using Streamlit
        stframe.image(frame, channels='RGB')
else:
    st.write('Stopped')
    cap.release()
