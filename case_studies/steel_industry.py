import os
import pandas as pd
import numpy as np
import sys
sys.path.append('./')
from StateCorr import StateCorr
from TSpy.utils import z_normalize

data_path = os.path.join(os.path.dirname(__file__), '../data/Steel_Industry/Steel_industry_data.csv')
df = pd.read_csv(data_path, sep=',', usecols=range(1,7))
data = np.expand_dims(df.to_numpy().T, axis=2)

print(data.shape)
sc = StateCorr()
state_seq_list = sc.fit_predict(data, 64, 20)
np.save('state_seq.npy', state_seq_list)

state_seq_list = np.load('state_seq.npy')

import matplotlib.pyplot as plt

plt.style.use('classic')
fig, ax = plt.subplots(nrows=6, sharex=True)

for i in range(6):
    ax[i].plot(z_normalize(data[i]))
    state_seq = state_seq_list[i].reshape(1,-1)
    state_seq = np.concatenate([state_seq, state_seq])
    ax[i].imshow(state_seq, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')

plt.tight_layout()
plt.show()