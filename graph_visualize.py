import streamlit as st
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import time
from model_test import extract_mfcc
import librosa
import soundfile as sf
import pandas as pd
import altair as alt
from scipy.signal import savgol_filter
import noisereduce as nr
import scipy.io.wavfile as wav
from pedalboard.io import AudioFile
from pedalboard import *
from tensorflow.image import resize
from pydub import AudioSegment
import io
import spotipy
import streamlit as st
from spotipy.oauth2 import SpotifyClientCredentials
import math

# Spotify Client
client = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id="1ed575a2d0954e1ba040771e1de31b12",
        client_secret="df52f67f61734867925bc2cdf3089930",
    ),
    # redirect_uri="localhost:8501",
)




# Song Searching
def search_song(genre_select, artist, year):
    result = {}
    st.write('after ==>',genre_select)
    keyword = f"genre={genre_select if (genre_select is not None) else ''}&artist={artist}&year={year[0]}-{year[1]}"
    st.write(keyword)
    # if genre_select is not None:
    try:
        result = client.search(q=keyword, type="track", limit=song_result)
        print("result ==>", result)
    except Exception as err:
        print(err)
    # elif emotion is not None:
    #     result = client.search(q=emotion, type="track", limit=3)
    # else:
    # st.warning('Please select the genre first!')

    # Get Tracks List
    # st.json(result and result)
    if result["tracks"]["items"]:
        tracks = []
        # st.json(result["tracks"]["items"])
        for track in result["tracks"]["items"]:
            tracks.append(
                {
                    "title": track["name"],
                    "href": track["external_urls"]["spotify"],
                    "artists": track["artists"],
                    "picture": track["album"]["images"][1]["url"],
                    "year": track["album"]["release_date"],
                    "preview_url": track["preview_url"],
                }
            )
        # st.json(tracks)

        grid = math.ceil(len(tracks) / 2)
        rows = [
            st.columns(2, vertical_alignment="bottom", gap="large") for i in range(grid)
        ]

        idx = 0
        for row in rows:
            for col in row:
                if idx >= len(tracks):
                    break

                with col:
                    st.image(tracks[idx]["picture"], width=100)

                    st.write(
                        '<h4><a target="_blank" href="{}" style="text-decoration:none;color: tomato">{}</a></h4>'.format(
                            tracks[idx]["href"], tracks[idx]["title"]
                        ),
                        unsafe_allow_html=True,
                    )
                    artists_name = []
                    for artist in tracks[idx]["artists"]:
                        artists_name.append(
                            '<a target="_blank" href="{}" style="text-decoration:none;color:gold">{}</a>'.format(
                                artist["external_urls"]["spotify"], artist["name"]
                            )
                        )
                    st.write(
                        " , ".join(artists_name),
                        unsafe_allow_html=True,
                    )

                    st.write(
                        f'<span style="color:lightSalmon">{tracks[idx]["year"][:4]}',
                        unsafe_allow_html=True,
                    )
                    if tracks[idx]["preview_url"]:
                        st.audio(tracks[idx]["preview_url"])
                    else:
                        st.warning("⚠️ Preview is not available")
                idx += 1
    else:
        st.markdown("#### No song for recommendation ")

# Define a callback function to collect audio data
recording = np.zeros(0)
def callback(indata, frames, time, status):
    global recording
    recording = np.append(recording, indata[:, 0])

def recoding_audio():

    st.write("Recording...")
    # duration_timer(duration)

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

    stream.stop()
    stream.close()
    st.write("Recording finished.")
    save_audio("scipy.wav", sample_rate, recording)


# Function to update the plot dynamically
def plot_waveform(data, sample_rate, fig):
    fig.add_rows(data)


# Load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    signal, sample_rate = librosa.load(file_path, sr=None)
    # audio_data = np.array(file_path)
    # sample_rate = 44100
    # Perform preprocessing (e.g., convert to Mel spectrogram and resize)
    # Define the duration of each chunk and overlap
    duration = len(signal) / sample_rate

    # Determine the start and end points for the middle 30 seconds
    start_time = (duration - 30) / 2
    end_time = start_time + 30

    # Convert time to sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Extract the middle 30 seconds of audio
    middle_audio = signal[start_sample:end_sample]
    st.audio(middle_audio, format="audio/mpeg", sample_rate=sample_rate)

    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds

    # Convert durations to samples
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    # Calculate the number of chunks
    num_chunks = (
        int(
            np.ceil(
                (len(middle_audio) - chunk_samples) / (chunk_samples - overlap_samples)
            )
        )
        + 1
    )

    # Iterate over each chunk
    for i in range(num_chunks):
        # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples

        # Extract the chunk of audio
        chunk = middle_audio[start:end]

        # Compute the Mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)

        # mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)


st.title("Waveform Visualization (Test)")

sample_rate = 44100  # samples per second
# sample_rate = 22050  # samples per second
buffer_size = 1024  # buffer size for real-time plotting
# file_path = "../mfcc_genre_detection/datasets/metal/metal.00069.wav"
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
genre_detection_result = None


def save_audio(filename, sample_rate, data):
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    wav.write(filename, sample_rate, (data * 32767).astype(np.int16))


def duration_timer(duration):
    with st.empty():
        for s in range(duration):
            second = duration - s if duration - s >= 10 else "0" + str(duration - s)
            st.write(f"⏳ 00:{second} seconds have passed")
            time.sleep(1)
        st.write(f"✔️ {duration}s music has recorded")


def model_service(model_path):
    st.write("Loading model... 🔃.")
    model = load_model(model_path)

    st.write("Extract MFCC... ✨.")

    if model_path == "./model/cnn__genre_detection_44100hz_0.91.h5":
        val = extract_mfcc(file_path="scipy.wav")
        val = np.array(val)
        val = val[..., np.newaxis]
    else:
        val = load_and_preprocess_data(file_path="scipy.wav")

    data = pd.DataFrame({"genres": genres_list, "values": list(genres.values())})
    st.write("Prediction starting... 🚀")
    prediction = model.predict(val)
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
            # genres[genres_list[idx]] = genres[genres_list[idx]] + v
            genres[genres_list[idx]] = ((genres[genres_list[idx]] * 3) + v) / 4
            fig.add_rows(
                [
                    {
                        "genres": genres_list[idx],
                        "values": round(v * 100),
                    }
                ]
            )


def model_result():
    st.write("<<========= Genre Detection Accuracy =======>>")
    # Sort the two lists together based on values in descending order
    global genres
    st.write(genres)
    genre_values = list(genres.values())
    st.write(math.fsum(genre_values))
    sorted_genres = sorted(
        zip(genres_list, genre_values), key=lambda x: x[1], reverse=True
    )
    # st.write('sorted_genres ==> ', sorted_genres)
    global genre_detection_result
    for i, name in enumerate(genres_list):
        if name == sorted_genres[0][0]:
            st.write("genre detection result :", i)
            genre_detection_result = i
            break
    # Print the top 3 genres and their values
    val = 0
    for genre, value in sorted_genres[:3]:
        st.write("{}\t\t==> {}%".format(genre, round(value * 100, 2)))
        val += float(value)

    st.write(val)
    st.write(f"Others : {round((1  - val) * 100 , 2) }%")

    st.write("Prediction finished. 👌")


# emotion = st.selectbox(
#     "Pick one the emotion you feel now",
#     ["happy", "sad", "surprise", "love", "neutral", "angry", "fear"],
# )
artist = st.text_input("Artist name")
year = st.slider("Song Release", 1980, 2024, [2009, 2017])
st.write(f"{year[0]}-{year[1]}")
# popularity = st.slider("Choose song popularity", 0, 100)
# st.markdown("*100 is most populer*")
song_result = st.slider("Song result", 3, 20)

model_path = st.radio(
    "Select the model",
    [
        "./model/cnn__genre_detection_44100hz_0.91.h5",
        "./model/cnn__genre_detection_41100hz_0.95(Tripathi Dataset).h5",
    ],
    captions=["Model accuracy: 91%", "Model accuracy: 95% (Tripathi Dataset)"],
)

music_input_method = st.radio(
    "Select the method of music input",
    ["Record music 🎙️", "Upload music file 🎵"],
    captions=["Record music sample around you", "Upload sample music from your device"],
)



genre_select = None
button_search = None
if music_input_method == "Record music 🎙️":
    duration = st.slider("Select recording duration (seconds)", 0, 30, 30, 5)
    recording_btn = st.button("Start Recording ⏺️")
    if recording_btn:
        recoding_audio()
        model_service(model_path)
        model_result()

else:
    uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
    if uploaded_file is not None:
        audio_data = uploaded_file.getvalue()
        data, samplerate = sf.read(io.BytesIO(audio_data))
        sf.write("scipy.wav", data, samplerate)
        model_service(model_path)
        model_result()
        genre_select = st.selectbox(
            "Music genre to search",
            [
                "blues",
                "classical",
                "country",
                "disco",
                "hip-hop",
                "jazz",
                "metal",
                "pop",
                "reggae",
                "rock",
            ],
            index=genre_detection_result,
        )
        st.write('before ==>',genre_select)

        button_search = st.button("Search Song")

genre_select = st.selectbox(
    "Music genre to search",
    [
        "blues",
        "classical",
        "country",
        "disco",
        "hip-hop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ],
    index=genre_detection_result,
)
st.write('before ==>',genre_select)

button_search = st.button("Search Song")
if button_search:
    search_song(genre_select, artist, year)