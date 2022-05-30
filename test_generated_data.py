from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
import os
import pickle
import torch
from Speech2Text import s2t

files = os.listdir('speech2text/created_data')
i = 0
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
#model = s2t()
for file in files:

    path = 'speech2text/created_data/' + file
    with open(path, 'rb') as fp:
        itemlist = pickle.load(fp)

    transcription = processor.batch_decode(itemlist[0]['labels'])
    print(transcription[0])
    input_values = itemlist[0]['input_values']
    logits = model(input_values).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    print(transcription[0])




