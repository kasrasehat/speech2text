from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
from torch.optim.lr_scheduler import StepLR
import torch
import librosa
import pickle
import pydub
import numpy as np



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

# load pretrained model and define optimizer
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 4e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
model.freeze_feature_encoder()
model.train()

with open("speech2text/pharmacy6.pickle", 'rb') as output:
  data = pickle.load(output)

l = 0
x_train = []
y_train = []
# load audio and train
for drug in data.keys():

    l += 1
    for j in range(len(data[drug]['brand_audio'])):

        file = data[drug]['brand_audio'][j]['path']
        path = 'speech2text' + file[1:]

        if path[-3:] == 'wav':

          try:
              speech, _ = librosa.load(path, sr=16000)

          except: continue

        elif path[-3:] == 'mp3':

          try:
              _, speech = read(path, normalized=True)
          except: continue

        else: continue

        input_values = processor(speech, sampling_rate=_, return_tensors="pt").input_values
        target_transcription = drug

    # encode labels
        with processor.as_target_processor():
          labels = processor(target_transcription, return_tensors="pt").input_ids

        x_train.append(input_values)
        y_train.append(labels)
