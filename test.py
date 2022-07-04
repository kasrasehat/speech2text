
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
from fuzzywuzzy import fuzz

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


def return_rxnorm(transcript, word2num):

    last_number = 'aaaa'
    flag = False
    for word in word2num.keys():
        if fuzz.ratio(transcript.split()[-1], word.upper()) >= 80 and fuzz.ratio(transcript.split()[-1], word.upper()) > fuzz.ratio(transcript.split()[-1], last_number):
            last_number = word
            flag = True

    if flag:
        return transcript.replace(' ' + transcript.split()[-1], '_' + word2num[last_number])
    else:
        return transcript


if __name__ == "__main__":

    num2word = {'9': 'noh', '15': 'panezdah', '20': 'bist', '30': 'see', '40': 'chehel', '50': 'panjah', '60': 'shast',
                '70': 'haftad',
                '75': 'haftadopanj', '80': 'hashtad', '85': 'hashtadopanj', '90': 'navad', '95': 'navadopanj',
                '100': 'sad',
                '110': 'sadodah', '120': 'sadobist', '125': 'sadobistopanj', '130': 'sadosee', '135': 'sadosiopanj',
                '150': 'sadopanjah',
                '155': 'sadopanjahopanj', '160': 'sadoshast', '165': 'sadoshastopanj', '180': 'sadohashtad',
                '190': 'sadonavad', '200': 'devist',
                '210': 'devistodah', '225': 'devistobistopanj', '240': 'devistochehel', '250': 'devistopanjah',
                '300': 'seesad', '325': 'sisadobistopanj',
                '320': 'sisadobist', '350': 'sisadopanjah', '375': 'sisadohaftadopanj', '400': 'chaharsad',
                '380': 'sisadohashtad', '450': 'chaharsadopanjah',
                '500': 'pansad', '550': 'pansadopanjah', '600': 'shishsad', '650': 'shishsadopanjah',
                '750': 'haftsadopanjah', '800': 'hashtsad',
                '833': 'hashtsadosiose', '999': 'nohsadonavadonoh', '1000': 'hezar', '1500': 'hezaropansad',
                '2000': 'dohezar', '2500': 'dohezaropansad',
                '2400': 'dohezarochaharsad', '3000': 'sehezar', '4000': 'chaharhezar', '5000': 'panjhezar',
                '5600': 'panjhezaroshishsad',
                '6000': 'sheshhezar', '10000': 'dahhezar', '12500': 'davazdahhezaropansad', '15000': 'panezdahhezar',
                '20000': 'bisthezar',
                '25000': 'bistopanjhezar', '50000': 'panjahhezar', '100000': 'sadhezar', '500000': 'pansadhezar',
                '1000000': 'yekmilion',
                '1500000': 'yekmilionopansadhezar'}

    word2num = {}
    for num in num2word.keys():
        word2num[num2word[num]] = num

    device = torch.device("cuda:0")
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    myload = torch.load("saved_models/30layers_augment_dosage_loss_7.7")
    try:
        model.load_state_dict(myload['state_dict'])
    except:
        model.load_state_dict(myload)

    # audio file is decoded on the fly
    path = 'speech2text/created_combined_data/acetaminophen_325(8).wav'
    speech, _ = librosa.load(path, sr=16000)
    #_, speech = read(path, normalized=True)
    inputs = processor(speech, sampling_rate=_, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    # transcribe speech
    transcription = processor.batch_decode(predicted_ids)
    print(return_rxnorm(transcription[0], word2num))
    #print(" ".join(processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())))