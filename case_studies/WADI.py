import os
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
sys.path.append('./')
from StateCorr import StateCorr
from TSpy.utils import z_normalize

USE_COMPUTED_STATE_SEQ = True

"""
Load data
"""
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

# data = data[20:40]
sc = StateCorr()

"""
Use StateCorr
"""
if USE_COMPUTED_STATE_SEQ:
    state_seq_list = np.load('state_seq.npy')
    sc.load_state_seq_list(state_seq_list)
    corr_matrix = np.load('corr_matrix.npy')
else:
    state_seq_list = sc.fit_predict(data, 512, 100).state_seq_list
    np.save('state_seq.npy', state_seq_list)
    corr_matrix = sc.calculate_matrix().corr_matrix
    np.save('corr_matrix.npy', corr_matrix)

"""
Show Results
"""
def find_top_k(id, k, matrix, state_seq_list):
    col = matrix[:,id]
    idx = np.argsort(-col)
    print(idx)

    plt.style.use('classic')
    fig, ax = plt.subplots(nrows=k, sharex=True, figsize=(8,4.5))
    for i in range(k):
        print(idx[i])
        data_ = z_normalize(data[idx[i]])

        state_seq = state_seq_list[idx[i]].reshape(1,-1)
        state_seq = np.concatenate([state_seq, state_seq])
        ax[i].imshow(state_seq, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.6, origin='lower', vmax=5, vmin=0)
        ax[i].plot(data_, lw=0.5)
        ax[i].set_ylim([-0.1,1.1])

        if i == 0:
            ax[i].set_title('Indicator %d'%(i+1), loc='left', y=0.55, x=0.02,
                fontsize='medium')
        elif i == 1:
            ax[i].set_title('%d-st state-correlated indicator'%(i), loc='left', y=0.55, x=0.02,
                fontsize='medium')
        elif i == 2:
            ax[i].set_title('%d-nd state-correlated indicator'%(i), loc='left', y=0.55, x=0.02,
                fontsize='medium')
        elif i == 3:
            ax[i].set_title('%d-rd state-correlated indicator'%(i), loc='left', y=0.55, x=0.02,
                fontsize='medium')
        else:
            ax[i].set_title('%d-th state-correlated indicator'%(i), loc='left', y=0.55, x=0.02,
                fontsize='medium')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.show()
    print(idx[:k])

find_top_k(21, 6, corr_matrix, state_seq_list)

#73 21
# find_top_k(18, 5)

# import matplotlib.pyplot as plt

# plt.style.use('classic')
# fig, ax = plt.subplots(nrows=20, sharex=True)

# for i in range(20):
#     ax[i].plot(z_normalize(data[i]))
#     state_seq = state_seq_list[i].reshape(1,-1)
#     state_seq = np.concatenate([state_seq, state_seq])
#     ax[i].imshow(state_seq, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')

# plt.tight_layout()
# plt.show()