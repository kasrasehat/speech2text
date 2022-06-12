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

l = 1810
k = 0
x_train = []
y_train = []
bug = {}
# load audio and train
for drug in list(data.keys())[l:]:


    print('{}% is completed'.format(100 * l / len(data.keys())))
    n_used_langs = []
    print('{}th drug is {} and the last file number is {}'.format(l, drug, k))
    l += 1
    target_transcription = drug.upper()
    # with processor.as_target_processor():
    #     a = processor(target_transcription, return_tensors="pt").input_ids

    #for langu in gtts.lang.tts_langs().keys():
    #gtts.lang.tts_langs().keys()
    g=0
    for langu in ['tr', 'it', 'en']:

        print(langu)
        g+=1

        try:
            tts = gtts.gTTS(drug, lang=langu)
            tts.save(f"speech2text/create_original_data/{drug}"+str(g)+".mp3")
        except:
            n_used_langs.append(langu)
            print(f'you have to stop run and restart from k = {k} and l={l-1}')
            break


        k += 1
        #_, speech = read("speech2text/temp.mp3", normalized=True)
        #os.remove("speech2text/temp.mp3")
        #input_values = processor(speech, sampling_rate=_, return_tensors="pt")
        #with torch.no_grad():
            #logits = model(**input_values).logits
        #predicted_ids = torch.argmax(logits, dim=-1)
        #print(" ".join(processor.tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())))
        #transcription = processor.decode(predicted_ids[0])
        #transcription = transcription.upper()
        #input_values['labels'] = a
        # loss = self.model(**input_values).loss
        # create a binary pickle file
        #path = 'speech2text/created_data/'+str(k)+'.pkl'
        # with open(path, 'wb') as fp:
        #     pickle.dump([input_values], fp)