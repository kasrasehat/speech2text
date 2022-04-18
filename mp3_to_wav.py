from os import path
from pydub import AudioSegment

# files
src = "voice/ACETAMINOPHEN2m"
dst = "voice/ACETAMINOPHEN4w.wav"

# convert wav to mp3
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")