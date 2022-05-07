from Speech2Text import s2t
import librosa
import pandas as pd
import numpy as np
import os
from ast import literal_eval
import pydub
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

pd.options.mode.chained_assignment = None

df = pd.read_csv('speech2text/map_dataset2excel.csv', converters={'feature': literal_eval})
df['hubert'] = pd.Series(np.nan, index=df.index)
df['fine_tuned_hubert'] = pd.Series(np.nan, index=df.index)
df['wave2vec_large'] = pd.Series(np.nan, index=df.index)
df['wave2vec_base'] = pd.Series(np.nan, index=df.index)
df['s2t_large'] = pd.Series(np.nan, index=df.index)
df['xlsr_multilingual'] = pd.Series(np.nan, index=df.index)

model = s2t()
p = -1

for i in tqdm.tqdm(range(df.shape[0])):
    p = p+1
    if type(df.iloc[i]['AudioPaths'])==str:
        hubert = []
        fine_tuned_hubert = []
        wave2vec_large = []
        wave2vec_base = []
        s2t_facebook = []
        xlsr_multilingual = []
        intractable = []

        for file in literal_eval(df.iloc[i]['AudioPaths']):
            path = 'speech2text/'+ file[2:]

            if path[-3:]=='wav':

                try:
                    speech, _ = librosa.load(path, sr=16000)
                except:
                    intractable.append(p)
                    continue

            elif path[-3:]=='mp3':

                try:
                    _,speech = read(path, normalized=True)
                except:
                    intractable.append(p)
                    continue

            else: continue

            hubert.append(model.HUBERT(speech))
            fine_tuned_hubert.append(model.fine_tuned_HUBERT(speech))
            wave2vec_large.append(model.Wave2Vec2_Large(speech))
            wave2vec_base.append(model.Wave2Vec2_Base(speech))
            s2t_facebook.append(model.facebook_s2t_large(speech))
            xlsr_multilingual.append(model.xlsr_multilingual_56(speech, lang_code='en'))

        df['hubert'][i] = hubert
        df['fine_tuned_hubert'][i] = fine_tuned_hubert
        df['wave2vec_large'][i] = wave2vec_large
        df['wave2vec_base'][i] = wave2vec_base
        df['s2t_large'][i] = s2t_facebook
        df['xlsr_multilingual'][i] = xlsr_multilingual


df.to_csv('final_file2.csv')









