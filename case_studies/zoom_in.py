from cProfile import label
import os
import numpy as np
import matplotlib.pyplot as plt
from TSpy.label import adjust_label
from TSpy.dataset import load_SMD

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/SMD/')

use_data = 'machine-1-6'

matrix_path = os.path.join(script_path, '../case_study1/state128/'+use_data+'.npy')
# matrix_path = os.path.join(script_path, '../case_study1/pearson/'+use_data+'.npy')
state_seq_path = os.path.join(script_path, '../case_study1/state_seq-128/'+use_data+'.npy')

def exclude_outlier(X):
    mean = np.mean(X)
    idx = np.argwhere(X>=3*mean)
    X[idx] = mean
    idx = np.argwhere(X<=mean/3)
    X[idx] = mean
    return X

def find_top_k(id, k):
    data,label,_,_,_,_ = load_SMD(data_path, use_data)
    corr_matrix = np.load(matrix_path)
    state_seq_array = np.load(state_seq_path)
    col = corr_matrix[:,id]
    idx = np.argsort(-col)
    print(idx)

    # fig, ax = plt.subplots(nrows=k, sharex=True, figsize=(8,4))
    # for i in range(k):
    data_ = data[:,idx[1]]
    state_seq = adjust_label(state_seq_array[idx[1]]).reshape(1,-1)
    state_seq = np.concatenate([state_seq, state_seq])
    label = np.concatenate([label.reshape(1,-1), label.reshape(1,-1)])
    print(label.shape)
    plt.figure(figsize=(4,1.5))
    plt.imshow(state_seq, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')
    # plt.imshow(label, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')
    plt.plot(data_, lw=0.5)
    plt.ylim([-0.1,1.1])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.savefig('zoom_in.png')
    print(idx[:k])

find_top_k(14, 5)