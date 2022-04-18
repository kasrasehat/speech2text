from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch
import librosa

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("voidful/wav2vec2-xlsr-multilingual-56")
model = Wav2Vec2ForCTC.from_pretrained("voidful/wav2vec2-xlsr-multilingual-56", from_tf = True)

# load dummy dataset and read soundfiles
path = 'FENTANYL CITRATE.wav'
speech, _ = librosa.load(path, sr = 16000)

# tokenize
input_values = processor(speech, sampling_rate=16_000,return_tensors="pt", padding="longest").input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(transcription)