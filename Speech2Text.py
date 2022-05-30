
import torch
from transformers import (Wav2Vec2ForCTC, Wav2Vec2Processor, HubertForCTC,
                        Speech2TextProcessor, Speech2TextForConditionalGeneration)
import pickle
from torch import nn

class s2t():

    def __init__(self):

        self.model1 = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        self.processor1 = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

        self.model2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        self.processor2 = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

        self.model3 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.processor3 = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        self.model4 = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr")
        self.processor4 = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")

        self.model5 = Wav2Vec2ForCTC.from_pretrained("voidful/wav2vec2-xlsr-multilingual-56")
        self.processor5 = Wav2Vec2Processor.from_pretrained("voidful/wav2vec2-xlsr-multilingual-56")

        self.model6 = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        self.processor6 = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

        myload = torch.load("saved_models/120layer_loss4.2")
        try:
            self.model6.load_state_dict(myload['state_dict'])
        except:
            self.model6.load_state_dict(myload)

       # myload = torch.load("saved_models/wave2vec2")
       # try:
       #     self.model3.load_state_dict(myload['state_dict'])
       # except:
       #     self.model3.load_state_dict(myload)


    def HUBERT(self, speech):

        inputs = self.processor1(speech, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
           logits = self.model1(**inputs).logits

        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor1.batch_decode(predicted_ids)
        return [transcription[0], log_probs]


    def Wave2Vec2_Large(self, speech):

        input_values = self.processor2(speech, sampling_rate=16_000,return_tensors="pt", padding="longest").input_values
        logits = self.model2(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor2.batch_decode(predicted_ids)
        return (transcription[0])


    def Wave2Vec2_Base(self, speech):

        input_values = self.processor3(speech, sampling_rate=16_000, return_tensors="pt",
                                 padding="longest").input_values
        logits = self.model3(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor3.batch_decode(predicted_ids)

        return transcription[0]


    def facebook_s2t_large(self, speech):
        input_features = self.processor4(
            speech,
            sampling_rate=16_000,
            return_tensors="pt"
        ).input_features  # Batch size 1
        generated_ids = self.model4.generate(input_features)
        return(self.processor4.batch_decode(generated_ids)[0])


    def xlsr_multilingual_56(self, speech, lang_code):

        features = self.processor5(speech, sampling_rate=16_000, padding=True, return_tensors="pt")
        input_values = features.input_values
        attention_mask = features.attention_mask

        with open("lang_ids.pk", 'rb') as output:
            lang_ids = pickle.load(output)

        with torch.no_grad():
            logits = self.model5(input_values, attention_mask=attention_mask).logits
            decoded_results = []

            for logit in logits:
                pred_ids = torch.argmax(logit, dim=-1)
                mask = ~pred_ids.eq(self.processor5.tokenizer.pad_token_id).unsqueeze(-1).expand(logit.size())
                vocab_size = logit.size()[-1]
                voice_prob = torch.nn.functional.softmax((torch.masked_select(logit, mask).view(-1, vocab_size)),
                                                         dim=-1)
                filtered_input = pred_ids[pred_ids != self.processor5.tokenizer.pad_token_id].view(1, -1)

                if len(filtered_input[0]) == 0:
                    decoded_results.append("")
                else:
                    lang_mask = torch.empty(voice_prob.shape[-1]).fill_(0)
                    lang_index = torch.tensor(sorted(lang_ids[lang_code]))
                    lang_mask.index_fill_(0, lang_index, 1)
                    lang_mask = lang_mask
                    comb_pred_ids = torch.argmax(lang_mask * voice_prob, dim=-1)
                    decoded_results.append(self.processor5.decode(comb_pred_ids))

        return decoded_results[0]


    def fine_tuned_HUBERT(self, speech):

        inputs = self.processor6(speech, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
           logits = self.model6(**inputs).logits

        log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor6.batch_decode(predicted_ids)
        return [transcription[0], log_probs]










