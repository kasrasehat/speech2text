from transformers import Wav2Vec2Processor, HubertForCTC
import torch
import librosa
import os
import pandas as pd
import pydub
import numpy as np
from fuzzywuzzy import fuzz
import regex as re
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
        if fuzz.ratio(transcript.split()[-1], word.upper()) >= 80 and fuzz.ratio(transcript.split()[-1], word.upper()) > fuzz.ratio(transcript.split()[-1], last_number.upper()):
            last_number = word
            flag = True

    if flag:
        return word2num[last_number]
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
    files = os.listdir('speech2text/Org2')
    tot = 0
    q = 0
    for file in files:

        path = 'speech2text/Org2' + '/' + file
        if path[-3:] == 'wav':

            try:
                speech1, _ = librosa.load(path, sr=16000)
            except:
                pass

        elif path[-3:] == 'mp3':

            try:
                _, speech1 = read(path, normalized=True)
                if speech1.shape[1] == 2:
                    speech1 = np.mean(speech1, axis=1)
            except:
                pass


        elif path[-3:] == 'm4a':

            try:
                speech1 = AudioSegment.from_file(path)
                path2 = path[:-4] + 'aaa.mp3'
                speech1.export(path2, format="mp3")
                _, speech1 = read(path2, normalized=True)
                if speech1.shape[1] == 2:
                    speech1 = np.mean(speech1, axis=1)
            except:
                pass

        else:
            continue

        label = re.findall('^\d+', file)
        if label[0] in num2word.keys():
            tot += 1
            #_, speech = read(path, normalized=True)
            inputs = processor(speech1, sampling_rate=_, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            # transcribe speech
            transcription = processor.batch_decode(predicted_ids)
            output = return_rxnorm(transcription[0], word2num)
            if output == label[0]:
                q += 1
            #print(" ".join(processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())))

    print(f'accuracy is {q/tot}')