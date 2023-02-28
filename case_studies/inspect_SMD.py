import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from TSpy.corr import cluster_corr
from TSpy.dataset import load_SMD
import tqdm

# metric = 'pearson'
# metric = 'spearman'
metric = 'state256'
# metric = 'state512'
# metric = 'state1024'

def exclude_outlier(X):
    mean = np.mean(X)
    idx = np.argwhere(X>=10*mean)
    X[idx] = mean
    return X

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/SMD/')
file_list = os.listdir(os.path.join(data_path, 'train'))

matrix_save_path = os.path.join(script_path, '../case_study1/'+metric)
fig_save_path = os.path.join(script_path, '../temp_fig/raw_'+metric)

if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)

for file_name in tqdm.tqdm(file_list):
    file_name = file_name[:-4] # remove '.txt'
    # use test data
    # _,_,_,_,data,label = load_SMD(data_path, file_name)
    # use all data
    data,label,_,_,_,_ = load_SMD(data_path, file_name)
    corr_matrix = np.load(os.path.join(matrix_save_path, file_name+'.npy'))
    corr_matrix, clustering_label, idx = cluster_corr(corr_matrix)
    fig, ax = plt.subplots(nrows=data.shape[1]+1, figsize=(16,32))
    for i in range(data.shape[1]):
        ax[i].plot(exclude_outlier(data[:,idx[i]]))
        # ax[i].plot(exclude_outlier(data[:,idx[i]]))
    ax[data.shape[1]].step(np.arange(len(label)), label, color='red')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(os.path.join(fig_save_path, file_name+'.png'))
    plt.close()