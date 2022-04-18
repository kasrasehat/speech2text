from transformers import Wav2Vec2Processor, HubertForCTC
import torch
import librosa

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

# audio file is decoded on the fly
path = 'voice/ACETAMINOPHEN2.mp3'
speech, _ = librosa.load(path, sr = 16000)
inputs = processor(speech, sampling_rate=_, return_tensors="pt")

with torch.no_grad():

    logits = model(**inputs).logits

predicted_ids = torch.argmax(logits, dim=-1)

# transcribe speech

transcription = processor.batch_decode(predicted_ids)

transcription[0]