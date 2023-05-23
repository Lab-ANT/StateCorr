import os
import numpy as np
import matplotlib.pyplot as plt
from TSpy.label import adjust_label, reorder_label
from TSpy.dataset import load_SMD

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/SMD/')

use_data = 'machine-1-6'

matrix_path = os.path.join(script_path, '../case_study1/state/'+use_data+'.npy')
# matrix_path = os.path.join(script_path, '../case_study1/pearson/'+use_data+'.npy')
state_seq_path = os.path.join(script_path, '../case_study1/state_seq/'+use_data+'.npy')

def exclude_outlier(X):
    mean = np.mean(X)
    idx = np.argwhere(X>=3*mean)
    X[idx] = mean
    idx = np.argwhere(X<=mean/3)
    X[idx] = mean
    return X

def plot_one(id):
    data,_,_,_,_,_ = load_SMD(data_path, use_data)
    corr_matrix = np.load(matrix_path)
    state_seq_array = np.load(state_seq_path)
    data_ = data[:,id]
    state_seq = reorder_label(adjust_label(state_seq_array[id])).reshape(1,-1)
    print(set(state_seq.flatten()))
    state_seq = np.concatenate([state_seq, state_seq])
    plt.figure(figsize=(8,2))
    plt.imshow(state_seq, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')
    plt.plot(data_, lw=0.5)
    plt.ylim([-0.1,1.1])
    plt.yticks([0,0.5,1],fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.savefig('machine.png')

plot_one(14)