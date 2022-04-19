from Speech2Text import s2t
import librosa
import pandas as pd
import numpy as np


model = s2t()
data = pd.read_csv('map_dataset2excel (1).csv')
for i in range(data.shape[0]):
    if not np.isnan():
        for path in data.iloc[i].AudioPaths:





