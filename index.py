import streamlit as st
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import time
import librosa
import os
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import time
from scipy.signal import savgol_filter

st.title("Live Audio Waveform Visualization")

sample_rate = 22050  # samples per second
buffer_size = 1024  # buffer size for real-time plotting
duration = 15
start_button = st.button("Start Recording 🎤")
# stop_button = st.button("Stop Recording ⏹️")

if start_button:
    st.write("Recording...")

    recording = np.zeros(0)

    # Define a callback function to collect audio data
    def callback(indata, frames, time, status):
        global recording
        recording = np.append(recording, indata[:, 0])

    # Create an input stream with the callback
    stream = sd.InputStream(
        callback=callback, channels=1, samplerate=sample_rate, blocksize=buffer_size
    )
    stream.start()

    # size = int(signal.size / 650)
    # df = []
    # for i in range(size):
    #     df.insert(i, signal[i * 650])

    # df = savgol_filter(df)
    fig = st.bar_chart()
    print(recording.size)
    start_time = time.time()
    recording_size = sample_rate * duration
    size = int(recording_size / 512)

    # while time.time() - start_time < duration:
    for i in range(size):
        time.sleep(0.0002)
        # print(countdown)
        step = int(i * 512)
        if step > recording.size:
            continue
        add_df = pd.DataFrame([recording[step]])
        fig.add_rows(add_df)

    # if stop_button:
    stream.stop()
    stream.close()
    st.write("Recording finished.")
