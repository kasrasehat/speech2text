from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import torch
import librosa

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# load dummy dataset and read soundfiles
path = 'voice/ENOXAPARIN.wav'
speech, _ = librosa.load(path, sr = 16000)

# tokenize
input_values = processor(speech, sampling_rate=16_000,return_tensors="pt", padding="longest").input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)