from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

import soundfile as sf
import torch
import librosa
import torchaudio
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


def predict_lang_specific(data, lang_code):
    features = processor(data, sampling_rate=16_000, padding=True, return_tensors="pt")
    input_values = features.input_values
    attention_mask = features.attention_mask
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
        decoded_results = []
        for logit in logits:
            pred_ids = torch.argmax(logit, dim=-1)
            mask = ~pred_ids.eq(processor.tokenizer.pad_token_id).unsqueeze(-1).expand(logit.size())
            vocab_size = logit.size()[-1]
            voice_prob = torch.nn.functional.softmax((torch.masked_select(logit, mask).view(-1, vocab_size)), dim=-1)
            filtered_input = pred_ids[pred_ids != processor.tokenizer.pad_token_id].view(1, -1)
            if len(filtered_input[0]) == 0:
                decoded_results.append("")
            else:
                lang_mask = torch.empty(voice_prob.shape[-1]).fill_(0)
                lang_index = torch.tensor(sorted(lang_ids[lang_code]))
                lang_mask.index_fill_(0, lang_index, 1)
                lang_mask = lang_mask
                comb_pred_ids = torch.argmax(lang_mask * voice_prob, dim=-1)
                decoded_results.append(processor.decode(comb_pred_ids))

    return decoded_results

model_name = "voidful/wav2vec2-xlsr-multilingual-56"
device = "cuda:0"
processor_name = "voidful/wav2vec2-xlsr-multilingual-56"

import pickle

with open("lang_ids.pk", 'rb') as output:
    lang_ids = pickle.load(output)

model = Wav2Vec2ForCTC.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(processor_name)



# load dummy dataset and read soundfiles
path = 'voice/ACETAMINOPHEN.wav'
speech, _ = librosa.load(path, sr = 16000)

# tokenize
features = processor(speech, sampling_rate=16_000,return_tensors="pt", padding=True)  # Batch size 1
input_values = features.input_values
attention_mask = features.attention_mask
with torch.no_grad():
    logit = model(input_values, attention_mask=attention_mask).logits
    decoded_results = []
    pred_ids = torch.argmax(logit, dim=-1)
    mask = pred_ids.ge(1).unsqueeze(-1).expand(logit.size())
    vocab_size = logit.size()[-1]
    voice_prob = torch.nn.functional.softmax((torch.masked_select(logit, mask).view(-1, vocab_size)), dim=-1)
    comb_pred_ids = torch.argmax(voice_prob, dim=-1)
    transcription = processor.decode(comb_pred_ids)

print(transcription)
print(predict_lang_specific(data=speech,lang_code='en')[0])
