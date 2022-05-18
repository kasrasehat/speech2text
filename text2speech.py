import gtts
from playsound import playsound
from io import BytesIO
import pydub
import numpy as np
import os


def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    a = a.set_frame_rate(16000)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y


tts = gtts.gTTS("alPRAZOLAM", lang='af')
tts.save("voice/ACETAMINOPHEN9.mp3")
_, speech = read("voice/ACETAMINOPHEN9.mp3", normalized=True)
#os.remove("voice/ACETAMINOPHEN9.mp3")

