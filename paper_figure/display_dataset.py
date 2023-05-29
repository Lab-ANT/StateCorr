import numpy as np
import os
from TSpy.corr import lagged_state_correlation, state_correlation
from TSpy.label import reorder_label
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

num_in_group = 5

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/synthetic_data/')
dataset_name = 'dataset5'
true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/')

plt.style.use('classic')
fig, ax = plt.subplots(nrows=5)
for i in range(num_in_group):
    data = np.load(os.path.join(data_path, '%s/test%d.npy'%(dataset_name, i)))
    state_seq = np.load(os.path.join(true_state_seq_path, 'state_seq_%s/test%d.npy'%(dataset_name, i)))
    print(data.shape)
    ax[i].plot(data, lw=0.8)
    ax[i].set_ylim([-1.5, 1.5])
    ax[i].set_yticks([])

    state_seq = state_seq.reshape(1,-1)
    state_seq = np.concatenate([state_seq, state_seq, state_seq])
    ax[i].imshow(state_seq, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.6, origin='lower', vmax=5, vmin=0)
    
    if i <4:
        ax[i].set_xticks([])

plt.tight_layout()
plt.savefig('dataset.png')