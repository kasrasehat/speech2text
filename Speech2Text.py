
import torch
from transformers import (Wav2Vec2ForCTC, Wav2Vec2Processor, HubertForCTC,
                        Speech2TextProcessor, Speech2TextForConditionalGeneration)


class s2t():

    def __init__(self):

        self.model1 = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        self.processor1 = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

        self.model2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        self.processor2 = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

        self.model3 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.processor3 = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

        self.model4 = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr")
        self.processor4 = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")


    def HUBERT(self, speech):

        inputs = self.processor1(speech, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
           logits = self.model1(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor1.batch_decode(predicted_ids)
        return transcription[0]


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







