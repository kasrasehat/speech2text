from Speech2Text import s2t
import librosa
import pandas as pd
import numpy as np
import pickle
from ast import literal_eval
import pydub
import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC

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

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
pd.options.mode.chained_assignment = None
path = "speech2text/pharmacy6.pickle"

with open(path, 'rb') as output:
    data = pickle.load(output)

q = 0
for drug in data.keys():
    for j in range(len(data[drug]['brand_audio'])):
        q += 1

df = pd.DataFrame(columns=['drug'], index=range(q))
df['drug_codes'] = pd.Series(np.nan, index=df.index)
df['hubert'] = pd.Series(np.nan, index=df.index)
df['hubert_codes'] = pd.Series(np.nan, index=df.index)
df['fine_tuned_hubert'] = pd.Series(np.nan, index=df.index)
df['fine_tuned_hubert_codes'] = pd.Series(np.nan, index=df.index)
df['wave2vec_large'] = pd.Series(np.nan, index=df.index)
df['wave2vec_base'] = pd.Series(np.nan, index=df.index)
df['s2t_large'] = pd.Series(np.nan, index=df.index)
df['xlsr_multilingual'] = pd.Series(np.nan, index=df.index)

model = s2t()
i = -1
bug = {}

for drug in data.keys():

    for j in range(len(data[drug]['brand_audio'])):

        i +=1
        file = data[drug]['brand_audio'][j]['path']
        path = 'speech2text' + file[1:]

        if path[-3:] == 'wav':

            try:
                speech, _ = librosa.load(path, sr=16000)

            except:
                bug[drug] = j
                continue

        elif path[-3:] == 'mp3':

            try:
                _, speech = read(path, normalized=True)
            except:
                bug[drug] = j
                continue

        else:
            bug[drug] = j
            continue

        target_transcription = drug.upper()
        df['drug'][i] = target_transcription
        with processor.as_target_processor():
            a = processor(target_transcription, return_tensors="pt").input_ids

        df['drug_codes'][i] = str(a.tolist()[0])

        df['hubert'][i] = model.HUBERT(speech)
        with processor.as_target_processor():
            b = processor(df['hubert'][i], return_tensors="pt").input_ids

        df['hubert_codes'][i] = str(b.tolist()[0])

        df['fine_tuned_hubert'][i] = model.fine_tuned_HUBERT(speech)
        with processor.as_target_processor():
            c = processor(df['fine_tuned_hubert'][i], return_tensors="pt").input_ids

        df['fine_tuned_hubert_codes'][i] = str(c.tolist()[0])

        df['wave2vec_large'][i] = model.Wave2Vec2_Large(speech)
        df['wave2vec_base'][i] = model.Wave2Vec2_Base(speech)
        df['s2t_large'][i] = model.facebook_s2t_large(speech).upper()
        df['xlsr_multilingual'][i] = model.xlsr_multilingual_56(speech, lang_code='en').upper()
        if i % 100 == 0:
            print('{}% completed'.format(i/q))



df.to_csv('final_file200.csv')















