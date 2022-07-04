from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
import tqdm
import os
import librosa
import pickle
import pydub
import numpy as np
import regex as re
import torch

num2word = {'9':'noh', '15':'panezdah', '20':'bist', '30':'see', '40':'chehel', '50':'panjah', '60':'shast', '70':'haftad',
            '75':'haftadopanj', '80':'hashtad', '85':'hashtadopanj', '90':'navad', '95':'navadopanj', '100':'sad',
            '110':'sadodah', '120':'sadobist', '125':'sadobistopanj', '130':'sadosee', '135':'sadosiopanj', '150':'sadopanjah',
            '155':'sadopanjahopanj', '160':'sadoshast', '165':'sadoshastopanj', '180':'sadohashtad', '190':'sadonavad', '200':'devist',
            '210':'devistodah', '225':'devistobistopanj', '240':'devistochehel', '250':'devistopanjah', '300':'seesad', '325':'sisadobistopanj',
            '320':'sisadobist', '350':'sisadopanjah', '375':'sisadohaftadopanj', '400':'chaharsad', '380':'sisadohashtad', '450':'chaharsadopanjah',
            '500':'pansad', '550':'pansadopanjah', '600':'shishsad', '650':'shishsadopanjah', '750':'haftsadopanjah', '800':'hashtsad',
            '833':'hashtsadosiose', '999':'nohsadonavadonoh', '1000':'hezar', '1500':'hezaropansad', '2000':'dohezar', '2500':'dohezaropansad',
            '2400':'dohezarochaharsad', '3000':'sehezar', '4000':'chaharhezar', '5000':'panjhezar', '5600':'panjhezaroshishsad',
            '6000':'sheshhezar', '10000':'dahhezar', '12500':'davazdahhezaropansad', '15000':'panezdahhezar', '20000':'bisthezar',
            '25000':'bistopanjhezar', '50000':'panjahhezar', '100000':'sadhezar', '500000':'pansadhezar', '1000000':'yekmilion',
            '1500000':'yekmilionopansadhezar'}

word2num = {}
for num in num2word.keys():
    word2num[num2word[num]] = num

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
#
# processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
# model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
# files = os.listdir('E:\codes_py\speech2text\speech2text\create_original_data')
# k = len(os.listdir('E:\codes_py\speech2text\speech2text\created_data'))
# for i in tqdm.tqdm(range(len(files))):
#
#     file = files[i]
#     path = 'E:\codes_py\speech2text\speech2text\create_original_data' + '\\' + file
#     if file[-3:] == 'wav':
#
#         try:
#             speech, _ = librosa.load(path, sr=16000)
#         except:
#             continue
#
#     elif file[-3:] == 'mp3':
#
#         try:
#             _, speech = read(path, normalized=True)
#         except:
#             continue
#
#     else:
#         continue
#
#     drug = file[:-5]
#     target_transcription = drug.upper()
#     with processor.as_target_processor():
#         a = processor(target_transcription, return_tensors="pt").input_ids
#
#     input_values = processor(speech, sampling_rate=_, return_tensors="pt")
#     # with torch.no_grad():
#     # logits = model(**input_values).logits
#     # predicted_ids = torch.argmax(logits, dim=-1)
#     # print(" ".join(processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())))
#     # transcription = processor.decode(predicted_ids[0])
#     # transcription = transcription.upper()
#     input_values['labels'] = a
#     # loss = self.model(**input_values).loss
#     # create a binary pickle file
#     path = 'speech2text/created_data/' + str(k) + '.pkl'
#     k += 1
#     with open(path, 'wb') as fp:
#         pickle.dump([input_values], fp)

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
files = os.listdir('E:\codes_py\speech2text\speech2text\Org2')
k = len(os.listdir('E:\codes_py\speech2text\speech2text\created_data'))
p = 0
T = 0
for i in tqdm.tqdm(range(len(files))):

    file = files[i]
    path = 'E:\codes_py\speech2text\speech2text\created_combined_data' + '\\' + file
    if file[-3:] == 'wav':

        try:
            speech, _ = librosa.load(path, sr=16000)
        except:
            continue

    elif file[-3:] == 'mp3':

        try:
            _, speech = read(path, normalized=True)
        except:
            continue

    else:
        continue
    try:
        drug = file[:-5]
        drug = drug.upper().replace('_', ' ')
        ff = re.findall('\d+', drug)[0].replace(' ','')
        target_transcription = drug.replace(ff, num2word[ff]).upper()
        T += 1
    except:
        print(drug)
        print(target_transcription)



    with processor.as_target_processor():
        a = processor(target_transcription, return_tensors="pt").input_ids

    input_values = processor(speech, sampling_rate=_, return_tensors="pt")
    # with torch.no_grad():
    #       logits = model(**input_values).logits
    #       predicted_ids = torch.argmax(logits, dim=-1)
    #       print(" ".join(processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())))
    #       transcription = processor.decode(predicted_ids[0])
    #       transcription = transcription.upper()
    input_values['labels'] = a
    # loss = self.model(**input_values).loss
    # create a binary pickle file
    path = 'speech2text/created_data/' + str(k) + '.pkl'
    k += 1
    #with open(path, 'wb') as fp:
    #     pickle.dump([input_values], fp)