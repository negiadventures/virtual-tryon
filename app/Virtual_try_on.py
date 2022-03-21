import cv2
import streamlit as st


st.title("Virtual Try-on")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)

while run:
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')
