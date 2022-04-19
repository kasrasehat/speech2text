from Speech2Text import s2t
import librosa

path = 'voice/ACETAMINOPHEN.wav'
speech, _ = librosa.load(path, sr = 16000)
model = s2t()
print(model.HUBERT(speech))
print(model.Wave2Vec2_Large(speech))
print(model.Wave2Vec2_Base(speech))
print(model.facebook_s2t_large(speech))
