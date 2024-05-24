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
from scipy.special import softmax
from scipy.signal import savgol_filter


# <<<====================//=================//===================>>
                        # Vanilla Waveform
# <<<====================//=================//===================>>
# Function to update the plot dynamically
def plot_waveform(data, sr):
    fig = plt.figure(figsize=(10, 4))
    librosa.display.waveshow(data, sr=sr, color="orange")
    st.pyplot(fig)
    # times = np.arange(len(data)) / sample_rate  # Convert samples to time
    # print(data)
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 4))
    ax.plot(data)
    # ax.set_ylim([-1, 1])
    ax.set_title("Live Audio Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    # st.pyplot(fig)
    plt.close(fig)


st.title("Live Audio Waveform Visualization (Testing)")

path = "disco.00073.wav"
signal, sr = librosa.load(path, sr=22050)
# # frame_rate = st.slider("Select recording duration (seconds)", 1, 10, 5)

# # plot_waveform(signal, sr)


# <<<====================//=================//===================>>
                        # Animate Waveform
# <<<====================//=================//===================>>
# df = pd.DataFrame(signal, columns=(["value"]))
# st.write(df)
# size = int(signal.size / 1000)
# data = signal[:size]
# # time = np.arange(data.size)
# # df = pd.DataFrame(list(zip(time, data)), columns=["time", "value"])
# df = pd.DataFrame(list(zip(data)), columns=["value"])
# st.write(df)
# fig2 = px.histogram(
#     df,
#     # x="time",
#     # y="value",
#     # color="value",
#     # range_y=[-1, 1],
#     animation_frame=df.index,
#     animation_group=df.index,
# )

# fig2.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 20
# fig2.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 5
# st.write(fig2)

# group_labels = ['Group 1']

# # Create distplot with custom bin_size
# fig = ff.create_distplot([np.random.randn(200)], group_labels=group_labels)
# fig = plt.figure(figsize=(10, 4))
# librosa.display.waveshow(signal, sr=sr, color="orange")

# <<<====================//=================//===================>>
                        # Animate Waveform With Filters
# <<<====================//=================//===================>>
fig, ax = plt.subplots()
ax.plot(signal)
ax.set_ylim([-1, 1])
ax.set_title("Live Audio Waveform")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
# Plot!
# st.plotly_chart(fig)


def moving_average(data, window_size):
    smoothed_data = np.zeros_like(data)
    for i in range(len(data)):
        if i < window_size:
            smoothed_data[i] = np.mean(data[: i + 1])
        else:
            smoothed_data[i] = np.mean(data[i - window_size + 1 : i + 1])
    return smoothed_data


def exponential_smoothing(data, alpha):
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]
    for i in range(1, len(data)):
        smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]
    return smoothed_data


size = int(signal.size / 650)

df = []
for i in range(size):
    df.insert(i, signal[i * 650])

df2 = exponential_smoothing(df, 1)
df3 = savgol_filter(df, 16, 10)
df4 = moving_average(df, 32)

fig = st.bar_chart()
fig2 = st.bar_chart()
fig3 = st.bar_chart()
fig4 = st.bar_chart()

for tick in range(size):
    time.sleep(0.0002)
    add_df = pd.DataFrame([signal[tick * 650]])
    add_df2 = pd.DataFrame([df2[tick]])
    add_df3 = pd.DataFrame([df3[tick]])
    add_df4 = pd.DataFrame([df4[tick]])
    fig.add_rows(add_df)
    fig2.add_rows(add_df2)
    fig3.add_rows(add_df3)
    fig4.add_rows(add_df4)


# <<<====================//=================//===================>>
                        # Vanilla Waveform from recorder
# <<<====================//=================//===================>>

# duration = st.slider("Select recording duration (seconds)", 1, 10, 5)
# sample_rate = 22050  # samples per second
# buffer_size = 1024  # buffer size for real-time plotting

# if st.button("Start Recording"):
#     st.write("Recording...")
#     recording = np.zeros(0)

#     # Define a callback function to collect audio data
#     def callback(indata, frames, time, status):
#         global recording
#         recording = np.append(recording, indata[:, 0])

#     # Create an input stream with the callback
#     stream = sd.InputStream(
#         callback=callback, channels=1, samplerate=sample_rate, blocksize=buffer_size
#     )
#     stream.start()

#     # Update plot dynamically
#     start_time = time.time()
#     while time.time() - start_time < duration:
#         if len(recording) > 0:
#             plot_waveform(
#                 recording[-sample_rate * duration :], duration
#             )  # Plot the most recent 'duration' seconds of audio
#         # time.sleep(0.005)  # Update every 100 ms

#     stream.stop()
#     stream.close()
#     st.write("Recording finished.")
