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

for drug in data.keys():

    for j in range(1):

        i +=1
        file = data[drug]['brand_audio'][j]['path']
        path = 'speech2text' + file[1:]

        if path[-3:] == 'wav':

            try:
                speech, _ = librosa.load(path, sr=16000)
            except:
                continue

        elif path[-3:] == 'mp3':

            try:
                _, speech = read(path, normalized=True)
            except:
                continue

        else:
            continue

    if drug in rx.keys():
        for num in rx[drug]:
            path = 'speech2text/Org2' + '/' + str(num) + '.m4a'

            if path[-3:] == 'wav':

                try:
                    speech1, _ = librosa.load(path, sr=16000)
                except:
                    pass

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
                    if speech1.shape[1] == 2:
                        speech1 = np.mean(speech1, axis=1)
                except:
                    pass

            else:
                pass

            try:
                s = np.concatenate((speech[:-1000], speech1[1500:]), axis=0)
                store_path = f'speech2text/created_combined_data/{drug}_{num}(2).wav'
                write(store_path, 16000, s)
            except:
                pass
    else:
        print(drug)