import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import random
import math
import time
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#encode all characters
whitelist = [chr(i) for i in range(65, 91)]
whitelist1 = [chr(i) for i in [32,60,62]]
whitelist.extend(whitelist1)
values = np.array(whitelist)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[5, :])])







