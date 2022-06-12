# import soundfile as sf
# import torch
# from datasets import load_dataset
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# import soundfile as sf
# import torch
# import librosa
# import pickle
#
# # load pretrained model
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
#
# with open("speech2text/pharmacy6.pickle", 'rb') as output:
#   data = pickle.load(output)
# # load audio
# path = 'voice/ACETAMINOPHEN.wav'
# speech, _ = librosa.load(path, sr = 16000)
#
# # pad input values and return pt tensor
# input_values = processor(speech, sampling_rate=_, return_tensors="pt").input_values
#
# # INFERENCE
#
# # retrieve logits & take argmax
# logits = model(input_values).logits
# predicted_ids = torch.argmax(logits, dim=-1)
#
# # transcribe
# transcription = processor.decode(predicted_ids[0])
#
# # FINE-TUNE
#
# target_transcription = "ACETAMINOPHEN"
#
# # encode labels
# with processor.as_target_processor():
#   labels = processor(target_transcription, return_tensors="pt").input_ids
#
# # compute loss by passing labels
# loss = model(input_values, labels=labels).loss
# loss.backward()
import random
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
import gtts
import os
import librosa
import pickle
import pydub
import torch

#from helper import _plot_signal_and_augmented_signal

# Python 3.8
# install matplotlib, librosa
# install python3-tk -> sudo apt install python3-tk


def add_white_noise(signal, noise_percentage_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise * noise_percentage_factor
    return augmented_signal


def time_stretch(signal, time_stretch_rate):
    """Time stretching implemented with librosa:
    https://librosa.org/doc/main/generated/librosa.effects.pitch_shift.html?highlight=pitch%20shift#librosa.effects.pitch_shift
    """
    return librosa.effects.time_stretch(signal, time_stretch_rate)


def pitch_scale(signal, sr, num_semitones):
    """Pitch scaling implemented with librosa:
    https://librosa.org/doc/main/generated/librosa.effects.pitch_shift.html?highlight=pitch%20shift#librosa.effects.pitch_shift
    """
    return librosa.effects.pitch_shift(signal, sr, num_semitones)

def random_gain(signal, min_factor=0.1, max_factor=0.12):
    gain_rate = random.uniform(min_factor, max_factor)
    augmented_signal = signal * gain_rate
    return augmented_signal


def invert_polarity(signal):
    return signal * -1

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    a = a.set_frame_rate(16000)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2 ** 15
    else:
        return a.frame_rate, y


if __name__ == "__main__":
     #signal, sr = librosa.load('E:\codes_py\speech2text\speech2text\create_original_data\myoclonus2.mp3', sr = 16000)
     sr, signal = read('E:\codes_py\speech2text\speech2text\create_original_data\myoclonus2.mp3', normalized=True)
     augmented_signal = random_gain(signal, 3, 4)
     sf.write("voice/augmented_audio1.wav", augmented_signal, sr)