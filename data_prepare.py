from Speech2Text import s2t
import librosa
import pandas as pd
import numpy as np
import os
from ast import literal_eval

pd.options.mode.chained_assignment = None

df = pd.read_csv('speech2text/map_dataset2excel.csv', converters={'feature': literal_eval})
df['hubert'] = pd.Series(np.nan, index=df.index)
df['wave2vec_large'] = pd.Series(np.nan, index=df.index)
df['wave2vec_base'] = pd.Series(np.nan, index=df.index)
df['s2t_large'] = pd.Series(np.nan, index=df.index)
model = s2t()

for i in range(df.shape[0]):
    if type(df.iloc[i]['AudioPaths'])==str:
        hubert = []
        wave2vec_large = []
        wave2vec_base = []
        s2t_facebook = []
        for file in literal_eval(df.iloc[i]['AudioPaths']):

            path = 'speech2text/'+file[2:]
            if path[-3:]=='wav':
                speech, _ = librosa.load(path, sr=16000)
            #else: speech,_ = a2n.audio_from_file(path)

            hubert.append(model.HUBERT(speech))
            wave2vec_large.append(model.Wave2Vec2_Large(speech))
            wave2vec_base.append(model.Wave2Vec2_Base(speech))
            s2t_facebook.append(model.facebook_s2t_large(speech))

        df['hubert'][i] = hubert
        df['wave2vec_large'][i] = wave2vec_large
        df['wave2vec_large'][i] = wave2vec_base
        df['s2t_large'][i] = s2t_facebook


df.to_csv('final_file.csv')









