import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
import random
from torch.optim.lr_scheduler import StepLR
import torch
import librosa
import pickle
import pydub
import tqdm
import numpy as np
import gc


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

batch_size = 8
p = 0
loss_tot = 0
epoch = 1

# load audio and train

for i in range(epoch):

    model_loss = 0
    q = 0
    l = 0
    t = 0
    for drug in data.keys():
        l += 1

        for j in range(len(data[drug]['brand_audio'])):

            p += 1
            q += 1
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
              except: continue

            else: continue


            input_values = processor(speech, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                logits = model(**input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])
            transcription = transcription.lower()
            target_transcription = drug

        # encode labels
            with processor.as_target_processor():
              input_values['labels'] = processor(target_transcription, return_tensors="pt").input_ids

        # compute loss by passing labels
            loss = model(**input_values).loss
            loss_tot += loss

            if p == batch_size:

                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()
                t += 1
                p = 0
                model_loss += loss_tot.item()
                loss_tot = 0
                print('epoch {}: loss in {}% progress is {}'.format(epoch, 100 * l/len(data.keys()), model_loss/q))
                if t % 10 == 1:
                    filename = 'E:/codes_py/speech2text/saved_models/model_epoch_{}_progress_{}'.format(epoch, t)
                    torch.save(model.state_dict(), filename)

                



    print('epoch {} average loss is {:.6f}'.format(epoch, model_loss/q))
    scheduler.step()

filename = 'E:/codes_py/speech2text/saved_models/hubert'
torch.save(model.state_dict(),filename)