from transformers import Wav2Vec2Processor, HubertForCTC
import torch
import librosa
import os
import pandas as pd

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

# audio file is decoded on the fly
path = 'voice/ACETAMINOPHEN.wav'
speech, _ = librosa.load(path, sr = 16000)
inputs = processor(speech, sampling_rate=_, return_tensors="pt")

with torch.no_grad():

    logits = model(**inputs).logits

predicted_ids = torch.argmax(logits, dim=-1)

# transcribe speech
transcription = processor.batch_decode(predicted_ids)
print(transcription[0])
print(" ".join(processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())))