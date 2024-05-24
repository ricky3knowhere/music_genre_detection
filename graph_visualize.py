import streamlit as st
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import time
from model_test import extract_mfcc
import librosa
import soundfile as sf

# Function to update the plot dynamically
def plot_waveform(data, sample_rate, fig):
    # fig, ax = plt.subplots()
    # times = np.arange(len(data)) / sample_rate  # Convert samples to time
    # ax.plot(times, data)
    # ax.set_ylim([-1, 1])
    # ax.set_title('Live Audio Waveform')
    # ax.set_xlabel('Time (seconds)')
    # ax.set_ylabel('Amplitude')
    # data = data / sample_rate
    fig.add_rows(data)
    # plt.close(fig)


st.title("Waveform Visualization (Test)")

duration = st.slider("Select recording duration (seconds)", 1, 30, 5)
sample_rate = 22050  # samples per second
buffer_size = 1024  # buffer size for real-time plotting

if st.button("Start"):
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

    fig = st.bar_chart()
    # Update plot dynamically
    start_time = time.time()
    while time.time() - start_time < duration:
        if len(recording) > 0:
            plot_waveform(
                recording[-duration:], sample_rate, fig
            )  # Plot the most recent 'duration' seconds of audio
        time.sleep(0.5)  # Update every 100 ms
    sf.write('music.wav',recording,samplerate=sample_rate)
    stream.stop()
    stream.close()
    st.write("Recording finished.")

    model = load_model("./model/cnn__genre_detection.h5")
    val = extract_mfcc(file_path=recording)
    # print(val)
    val = np.array(val)
    # print(val.shape)
    val = val[..., np.newaxis]
    # print(val.shape)
    prediction = model.predict(val)
    predicted_index = np.argmax(prediction, axis=1)
    st.markdown(
        "*mfcc_val* ==> {} \nprediction ==> {}\npredicted_index ==> {}".format(
            val.shape, prediction.shape, predicted_index
        )
    )
