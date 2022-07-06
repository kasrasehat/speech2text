from Speech2Text import s2t
import librosa
import pandas as pd
import numpy as np
import pickle
import torch
from ast import literal_eval
import pydub
import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
from scipy.io.wavfile import write
from pydub import AudioSegment

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

rx = {}
RX = pd.read_csv('speech2text/rxnorm_numbers (1).csv')
for i in range(len(RX)):

    if RX['RxnormName'][i] in rx.keys():
        rx[RX['RxnormName'][i]].append(RX['Numbers'][i])
    else:
        rx[RX['RxnormName'][i]] = [RX['Numbers'][i]]

pd.options.mode.chained_assignment = None
path = "speech2text/pharmacy6.pickle"

with open(path, 'rb') as output:
    data = pickle.load(output)

drugs = []
k = 0
for drug in data.keys():

    drugs.append(drug.upper())
    k += 1
    if drug in rx.keys():

        for i in rx[drug]:

            drug_new = drug.upper() + f'_{i}'
            drugs.append(drug_new.upper())
            k += 1

df = pd.DataFrame({'drug': drugs})
df.to_csv('speech2text/drug_list')

