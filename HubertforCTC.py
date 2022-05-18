from transformers import Wav2Vec2Processor, HubertForCTC
import torch
import librosa
import os
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz


def compare(word, dataset):
    for j in range(data.shape[0]):
        if not pd.isna(data.iloc[j].drug):
            dataset.loc[j,('score')] = fuzz.ratio(word, dataset.iloc[j].drug)

    dataset1 = dataset.sort_values('score', ascending=False, inplace=False)
    dataset1 = dataset1.reset_index(drop=True)
    return list(dataset1['drug'][0:5]), list(dataset1['drug'][0:4]), list(dataset1['drug'][0:3]), list(dataset1['drug'][0:2]), list(dataset1['drug'][0:1])

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
myload = torch.load("saved_models/acoustic_hubert_epoch_9")
try:
    model.load_state_dict(myload['state_dict'])
except:
    model.load_state_dict(myload)

# audio file is decoded on the fly
path = 'speech2text/pharmacy.umich.edu/dichlorphenamide.wav'
speech, _ = librosa.load(path, sr = 16000)
inputs = processor(speech, sampling_rate=_, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_ids = torch.argmax(logits, dim=-1)

# transcribe speech
transcription = processor.batch_decode(predicted_ids)
print(transcription[0])
print(" ".join(processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())))
data = pd.read_csv('final_file.csv')
data1 = data.copy()
data['score'] = pd.Series(np.nan, index=data.index, dtype=int)
best5, best4, best3, best2, best1 = compare(transcription[0], data)
print(best1)
print(best2)
print(best5)