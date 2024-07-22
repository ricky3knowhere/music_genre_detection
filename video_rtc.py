import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import threading
import time

# Face detection using OpenCV's pre-trained Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.start_time = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Check if 5 seconds have passed since the start of the video stream
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 5:
            st.session_state["stop"] = True

        return img

# Main function to run the Streamlit app
def main():
    st.title("Face Detection App")

    if "stop" not in st.session_state:
        st.session_state["stop"] = False

    if st.session_state["stop"]:
        st.warning("Camera has been turned off after 5 seconds.")
    else:
        webrtc_streamer(key="example", video_transformer_factory=FaceDetectionTransformer)

if __name__ == "__main__":
    main()
