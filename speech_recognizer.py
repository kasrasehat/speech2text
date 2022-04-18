import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset
import soundfile as sf

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

ds = load_dataset(
    "patrickvonplaten/librispeech_asr_dummy",
    "clean",
    split="validation"
)
ds = ds.map(map_to_array)

input_features = processor(
    ds["speech"][1],
    sampling_rate=16_000,
    return_tensors="pt"
).input_features  # Batch size 1
generated_ids = model.generate(input_features)
print(processor.batch_decode(generated_ids))