import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import tempfile
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
import soundfile as sf


# Load YAMNet model from TensorFlow Hub
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)

# Classes YAMNet was trained on
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
with open(class_map_path, 'r') as f:
    class_names = [line.strip().split(',')[2] for line in f.readlines()[1:]]

# Target class to detect
SIREN_CLASSES = ['Siren', 'Emergency vehicle (siren)']

def extract_audio_from_video(video_path):
    """
    Extract audio from a video file and save to temporary WAV file.
    """
    clip = VideoFileClip(video_path)
    audio = clip.audio

    if audio is None:
        raise ValueError("No audio track found in the video.")

    audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    audio.write_audiofile(audio_temp.name, verbose=False, logger=None)
    return audio_temp.name

def detect_siren_audio(video_path):
    try:
        audio_path = extract_audio_from_video(video_path)
    except ValueError as e:
        print("Warning:", e)
        return None  # No audio fallback

    # Load YAMNet model
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    sample_rate = 16000

    # Load audio
    wav_data, sr = sf.read(audio_path, dtype='float32')
    if sr != sample_rate:
        wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=sample_rate)

    # Run YAMNet inference
    scores, embeddings, spectrogram = yamnet_model(wav_data)
    scores_np = scores.numpy()
    class_scores = scores_np.mean(axis=0)

    # Load label map
    class_map_path = tf.keras.utils.get_file('yamnet_class_map.csv',
        'https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv')
    class_names = [line.split(',')[2].strip() for line in open(class_map_path).readlines()[1:]]

    top_class = class_names[np.argmax(class_scores)]
    return "siren" in top_class.lower()
