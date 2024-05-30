import numpy as np
import pandas as pd
import math
import librosa
import streamlit as st
import altair as alt
from tensorflow.keras.models import load_model
import time

SAMPLE_RATE = 22050
TRACK_DURATION = 30  # second
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def extract_mfcc(file_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    # dictionary to store mapping, labels, and MFCCs
    data = []
    print('num_segments ==>',num_segments)
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    # signal = np.array(file_path)

    for d in range(num_segments):
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(
            y=signal[start:finish],
            sr=SAMPLE_RATE,
            n_mfcc=num_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        mfcc = mfcc.T

        # store only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data.append(mfcc.tolist())

    return data

genres = {
    "Blues": 0,
    "Classical": 0,
    "Country": 0,
    "Disco": 0,
    "HipHop": 0,
    "Jazz": 0,
    "Metal": 0,
    "Pop": 0,
    "Reggae": 0,
    "Rock": 0,
}
genres_list = list(genres.keys())

file_path = "../mfcc_genre_detection/datasets/classical/classical.00010.wav"

model = load_model("./model/cnn__genre_detection.h5")
val = extract_mfcc(file_path=file_path)
prediction = model.predict(val)

data = pd.DataFrame({"genres": genres_list, "values": list(genres.values())})
st.write("Prediction starting... 🚀")
# st.write(data)
fig = st.altair_chart(
    alt.Chart(data)
    .mark_bar()
    .encode(x="values:Q", y="genres", tooltip="values", color="genres")
    .properties(height=700),
    use_container_width=True,
)

for i in range(prediction.shape[0]):
    time.sleep(0.6)
    # print(prediction[i][8])
    for idx, v in enumerate(prediction[i]):
        # print('val', genres['Reggae'])
        genres[genres_list[idx]] = genres[genres_list[idx]] + v

        fig.add_rows(
            [
                {
                    "genres": genres_list[idx],
                    "values": round(genres[genres_list[idx]] * 10),
                }
            ]
        )

st.write("<<========= Genre Detection Accuracy =======>>")
# st.write(genres)
total_val = 0
for genre in genres:
    st.write("{}\t\t==> {}%".format(genre, round(genres[genre] * 10, 2)))
    total_val = total_val + genres[genre]

st.write("total_val ==>", total_val)

st.write("Prediction finished. 👌")
