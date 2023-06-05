import os
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('./')
from StateCorr import StateCorr
from TSpy.utils import z_normalize

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/2019.npy')

data = np.load(data_path)
data[np.isnan(data)] = 1

unused_col = []

for i in range(data.shape[1]):
    value_set = set(data[:500000,i])
    if len(value_set) == 1:
        print(i)
        unused_col.append(i)

data = np.delete(data, unused_col, axis=1)

data = np.expand_dims(data[::20].T, axis=2)
print(data.shape)

sc = StateCorr()
state_seq_list = sc.fit_predict(data[:6], 128, 20)
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











# df = pd.DataFrame(data[:500000])

# fig, ax = plt.subplots(nrows=10, sharex=True)
# for i in range(10):
#     ax[i].plot(data[::20, i+10])
# plt.show()

# matrix = df.corr('pearson').to_numpy()
# idx = np.isnan(matrix)
# print(idx)
# print(matrix.shape)
# matrix,_,_ = cluster_corr(matrix)
# # plt.imshow(matrix, cmap='gray')
# plt.imshow(matrix)
# plt.colorbar()
# plt.show()