import pickle
from pydub import AudioSegment
import librosa
from Speech2Text import s2t
import librosa
import pandas as pd
import numpy as np
import pickle
import torch
from ast import literal_eval
import pydub
from scipy.io.wavfile import write
import tqdm

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    a = a.set_frame_rate(16000)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

path = 'speech2text/merckmanuals.com/acetaminophen.mp3'
# with open(path, 'rb') as output:
#     data = pickle.load(output)

file = 'speech2text/create_original_data/acetaminophen7.wav'
bug = {}
if path[-3:] == 'wav':

    try:
        speech, _ = librosa.load(path, sr=16000)
    except: pass

elif path[-3:] == 'mp3':

    try:
        _, speech = read(path, normalized=True)
    except:
        pass

else: pass

path = 'speech2text/number_voices/Org2/100.m4a'
if path[-3:] == 'wav':

    try:
        speech1, _ = librosa.load(path, sr=16000)
    except: pass

elif path[-3:] == 'mp3':

    try:
        _, speech1 = read(path, normalized=True)
    except:
        pass


elif path[-3:] == 'm4a':

    try:
        speech1 = AudioSegment.from_file(path)
        path2 = path[:-4]+'bbb.mp3'
        speech1.export(path2, format="mp3")
        _, speech1 = read(path2, normalized=True)
        speech1 = np.mean(speech1, axis=0)

    except:
        pass

else:
    pass

s = np.concatenate((speech[:-1000], speech1[10500:]), axis=0)
write('speech2text/new_file1.wav', 16000, s)