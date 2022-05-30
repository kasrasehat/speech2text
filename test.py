
import numpy as np
import os
import torch
import argparse
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from Data_preprocess import prepare_data
from torch.utils.data import TensorDataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
from torch.optim.lr_scheduler import StepLR
import torch
from transformers import Wav2Vec2Processor, HubertForCTC
import torch
import librosa
import os
import pandas as pd
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

def del_unk(word):

    while '<' in word:
        in1 = word.index('<')
        in2 = word.index('>') + 1
        if in1 == 0:
            word = word[in2:]
        else:
            word = word[:in1]+word[in2:]

    return word



if __name__ == "__main__":


    device = torch.device("cuda:0")
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    myload = torch.load("saved_models/30layer-loss4.5")
    try:
        model.load_state_dict(myload['state_dict'])
    except:
        model.load_state_dict(myload)

    # audio file is decoded on the fly
    path = 'voice/PROSTAGLANDIN.wav'
    speech, _ = librosa.load(path, sr=16000)
    #_, speech = read(path, normalized=True)
    inputs = processor(speech, sampling_rate=_, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    # transcribe speech
    transcription = processor.batch_decode(predicted_ids)
    print(transcription[0])
    #print(" ".join(processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())))