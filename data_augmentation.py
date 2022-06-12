import random
import numpy as np
import soundfile as sf
import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
import gtts
import os
import librosa
import pickle
import pydub


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
    files = os.listdir('E:\codes_py\speech2text\speech2text\create_original_data')
    q = 0
    for i in tqdm.tqdm(range(len(files))):

        try:
            file = files[i]
            path = 'E:\codes_py\speech2text\speech2text\create_original_data' + '\\' + file
            sr, signal = read(path, normalized=True)

            augmented_signal1 = time_stretch(signal, 0.6)
            path1 = path[:-5] + '4.wav'
            sf.write(path1, augmented_signal1, sr)

            augmented_signal2 = time_stretch(signal, 1.4)
            path2 = path[:-5] + '5.wav'
            sf.write(path2, augmented_signal2, sr)

            augmented_signal3 = add_white_noise(signal, 0.6)
            path3 = path[:-5] + '6.wav'
            sf.write(path3, augmented_signal3, sr)

            augmented_signal4 = add_white_noise(signal, 0.3)
            augmented_signal4 = time_stretch(augmented_signal4, 0.7)
            path4 = path[:-5] + '7.wav'
            sf.write(path4, augmented_signal4, sr)

            augmented_signal5 = add_white_noise(signal, 0.3)
            augmented_signal5 = time_stretch(augmented_signal5, 1.3)
            path5 = path[:-5] + '8.wav'
            sf.write(path5, augmented_signal5, sr)

        except:
            q += 1
            print(q)


