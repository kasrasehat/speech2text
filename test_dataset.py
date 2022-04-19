# This is a sample Python script.

from datasets import load_dataset, load_metric
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor
import soundfile as sf

librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")  # change to "other" for other test dataset
wer = load_metric("wer")

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr").to("cuda")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr", do_upper_case=True)

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

librispeech_eval = librispeech_eval.map(map_to_array)

def map_to_pred(batch):
    features = processor(batch["speech"], sampling_rate=16000, padding=True, return_tensors="pt")
    input_features = features.input_features.to("cuda")
    attention_mask = features.attention_mask.to("cuda")

    gen_tokens = model.generate(input_ids=input_features, attention_mask=attention_mask)
    batch["transcription"] = processor.batch_decode(gen_tokens, skip_special_tokens=True)
    return batch

result = librispeech_eval.map(map_to_pred, batched=True, batch_size=8, remove_columns=["speech"])

print("WER:", wer(predictions=result["transcription"], references=result["text"]))
