import torch
import librosa
from transformers import Wav2Vec2Processor, HubertModel

from datasets import load_dataset

import soundfile as sf


processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

# load dummy dataset and read soundfiles
path = 'voice/ACETAMINOPHEN.wav'
speech, _ = librosa.load(path, sr = 16000)

# tokenize
input_values = processor(speech, sampling_rate=16_000,return_tensors="pt", padding="longest").input_values  # Batch size 1
hidden_states = model(input_values).last_hidden_state
feature = torch.argmax(hidden_states, dim=-1)
#print(hidden_states)

path = 'voice/ACETAMINOPHEN.wav'
speech, _ = librosa.load(path, sr = 16000)

# tokenize
input_values = processor(speech, sampling_rate=16_000,return_tensors="pt", padding="longest").input_values  # Batch size 1
hidden_states1 = model(input_values).last_hidden_state
feature1 = torch.argmax(hidden_states1, dim=-1)