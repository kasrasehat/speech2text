#%%
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
from torch.optim.lr_scheduler import StepLR
import torch
import librosa
import pickle
import pydub
import numpy as np


class prepare_data():

    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")


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


    def prep_data(self, path):

        with open(path, 'rb') as output:
          data = pickle.load(output)

        l = 0
        x_train = []
        y_train = []
        bug = {}
        # load audio and train
        for drug in data.keys():

            l += 1
            for j in range(len(data[drug]['brand_audio'])):

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
                      _, speech = self.read(path, normalized=True)
                  except:
                      bug[drug] = j
                      continue

                else:
                    bug[drug] = j
                    continue

                input_values = self.processor(speech, sampling_rate=_, return_tensors="pt")
                target_transcription = drug.upper()

            # encode labels
                with self.processor.as_target_processor():
                  input_values['labels'] = self.processor(target_transcription, return_tensors="pt").input_ids

                x_train.append(input_values)

        return x_train, bug