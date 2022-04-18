from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch
import librosa
import torchaudio
from datasets import load_dataset, load_metric
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    AutoTokenizer,
    AutoModelWithLMHead
)
import torch
import re
import sys
import soundfile as sf

model_name = "voidful/wav2vec2-xlsr-multilingual-56"
device = "cuda:0"
processor_name = "voidful/wav2vec2-xlsr-multilingual-56"

import pickle

with open("lang_ids.pk", 'rb') as output:
    lang_ids = pickle.load(output)

model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(processor_name)

model.eval()

# load dummy dataset and read soundfiles
path = 'ALUMINIUM.wav'
speech, _ = librosa.load(path, sr = 16000)

# tokenize
input_values = processor(speech, sampling_rate=16_000,return_tensors="pt", padding="longest").input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(transcription)