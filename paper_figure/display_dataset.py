import numpy as np
import os
import matplotlib.pyplot as plt
from TSpy.utils import z_normalize

num_in_group = 5

def find_cp_from_state_seq(X):
    pre = X[0]
    cp_list = []
    for i, e in enumerate(X):
        if e == pre:
            continue
        else:
            cp_list.append(i)
            pre = e
    return cp_list

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/synthetic_data/')
dataset_name = 'dataset3'
true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/')

plt.style.use('classic')
fig, ax = plt.subplots(nrows=5, figsize=(8,5))
for i in range(num_in_group):
    data = np.load(os.path.join(data_path, '%s/test%d.npy'%(dataset_name, i)))
    state_seq = np.load(os.path.join(true_state_seq_path, 'state_seq_%s/test%d.npy'%(dataset_name, i)))
    print(data.shape)
    data = z_normalize(data)
    ax[i].plot(data+0.8, lw=0.8)
    ax[i].set_ylim([0.3, 2])
    ax[i].set_yticks([])
    ax[i].set_xlim([0,20000])

    state_seq = state_seq.reshape(1,-1)
    # state_seq = np.concatenate([np.zeros(state_seq.shape), state_seq])
    ax[i].imshow(state_seq, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.6, origin='lower', vmax=5, vmin=0)
    
    state_seq = state_seq.flatten()
    cp_list = find_cp_from_state_seq(state_seq)
    # print(cp_list)
    for cp in cp_list:
            ax[i].vlines(cp, 0, 2, color="black")
    if i <4:
        ax[i].set_xticks([])

plt.tight_layout()
plt.savefig('dataset.png')