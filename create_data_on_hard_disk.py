from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC
import gtts
import os
import librosa
import pickle
import pydub
import numpy as np
import torch


processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    a = a.set_frame_rate(16000)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2 ** 15
    else:
        return a.frame_rate, y


path = "speech2text/pharmacy6.pickle"
with open(path, 'rb') as output:
  data = pickle.load(output)

l = 0
k = 0
x_train = []
y_train = []
bug = {}
# load audio and train
for drug in data.keys():

    print('{}% is completed'.format(100 * l / len(data.keys())))
    l += 1
    n_used_langs = []
    #for langu in gtts.lang.tts_langs().keys():
    #gtts.lang.tts_langs().keys()
    for langu in ['af', 'ar', 'fr', 'uk', 'tr', 'it', 'en']:

        print(langu)
        try:
            tts = gtts.gTTS(drug, lang=langu)
            tts.save("voice/temporary.mp3")
        except:
            n_used_langs.append(langu)
            continue

        _, speech = read("voice/temporary.mp3", normalized=True)
        os.remove("voice/temporary.mp3")

        input_values = processor(speech, sampling_rate=_, return_tensors="pt")
        #with torch.no_grad():
            #logits = model(**input_values).logits
        #predicted_ids = torch.argmax(logits, dim=-1)
        #print(" ".join(processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())))
        #transcription = processor.decode(predicted_ids[0])
        #transcription = transcription.upper()
        target_transcription = drug.upper()

        # encode labels
        with processor.as_target_processor():
            input_values['labels'] = processor(target_transcription, return_tensors="pt").input_ids

        # loss = self.model(**input_values).loss
        # create a binary pickle file
        path = 'speech2text/augmented_data/'+ str(k)+'.pkl'
        k +=1
        with open(path, 'wb') as fp:
            pickle.dump([input_values], fp)



