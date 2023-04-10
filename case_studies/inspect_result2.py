import os
import numpy as np
import matplotlib.pyplot as plt
from TSpy.label import adjust_label, reorder_label
from TSpy.dataset import load_SMD
# from TSpy.corr import match_label

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/SMD/')

use_data = 'machine-1-6'

# matrix_path = os.path.join(script_path, '../case_study1/pearson/'+use_data+'.npy')
matrix_path = os.path.join(script_path, '../case_study1/state128/'+use_data+'.npy')
state_seq_path = os.path.join(script_path, '../case_study1/state_seq-128/'+use_data+'.npy')

def exclude_outlier(X):
    mean = np.mean(X)
    idx = np.argwhere(X>=3*mean)
    X[idx] = mean
    idx = np.argwhere(X<=mean/3)
    X[idx] = mean
    return X

def find_top_k(id, k):
    data,_,_,_,_,_ = load_SMD(data_path, use_data)
    corr_matrix = np.load(matrix_path)
    state_seq_array = np.load(state_seq_path)
    col = corr_matrix[:,id]
    idx = np.argsort(-col)
    print(idx)

    # plt.style.use('bmh')
    state_seq1 = reorder_label(adjust_label(state_seq_array[idx[0]])).flatten()
    fig, ax = plt.subplots(nrows=k, sharex=True, figsize=(4,8))
    for i in range(k):
        data_ = data[:,idx[i]]
        # state_seq = match_label(state_seq1, reorder_label(adjust_label(state_seq_array[idx[i]]))).reshape(1,-1)
        state_seq = reorder_label(state_seq_array[idx[i]]).reshape(1,-1)
        state_seq = np.concatenate([state_seq, state_seq])
        # ax[i].plot(exclude_outlier(data_), color= '#348ABC')
        ax[i].imshow(state_seq, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')
        # ax[i].plot(data_, lw=0.5, color= '#348ABC')
        ax[i].plot(data_, lw=0.5)
        ax[i].set_ylim([-0.1,1.1])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.savefig('look.png')
    print(idx[:k])

find_top_k(14, 5)
# find_top_k(18, 5)