import json
import os
import math
import librosa
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("./model/cnn__genre_detection.h5")
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # second
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
# file_path = "../mfcc_genre_detection/datasets/disco/disco.00074.wav"


def extract_mfcc(file_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    # dictionary to store mapping, labels, and MFCCs
    data = []
    print('num_segments ==>',num_segments)
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    signal = np.array(file_path)

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


# val = extract_mfcc(file_path=file_path, num_segments=10)
# # print(val)
# val = np.array(val)
# # print(val.shape)
# val = val[..., np.newaxis]
# # print(val.shape)
# prediction = model.predict(val)
# predicted_index = np.argmax(prediction, axis=1)
# print(
#     "mfcc_val ==> {} \nprediction ==> {}\npredicted_index ==> {}".format(
#         val.shape, prediction.shape, predicted_index
#     )
# )