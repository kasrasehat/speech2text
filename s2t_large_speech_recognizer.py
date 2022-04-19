import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import soundfile as sf
import librosa


model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")

path = 'voice/ACETAMINOPHEN.wav'
speech, _ = librosa.load(path, sr = 16000)
input_features = processor(
    speech,
    sampling_rate=16_000,
    return_tensors="pt"
).input_features  # Batch size 1
generated_ids = model.generate(input_features)
print(processor.batch_decode(generated_ids)[0])

