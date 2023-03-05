import os
import numpy as np
import matplotlib.pyplot as plt
from TSpy.corr import cluster_corr

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/SMD/')

use_data = 'machine-1-1'

matrix_path1 = os.path.join(script_path, '../case_study1/state1024/'+use_data+'.npy')
matrix_path2 = os.path.join(script_path, '../case_study1/pearson/'+use_data+'.npy')
matrix_path3 = os.path.join(script_path, '../case_study1/spearman/'+use_data+'.npy')

def plot_one():
    corr_matrix_state,_,_ = cluster_corr(np.load(matrix_path1))
    corr_matrix_pearson,_,_ = cluster_corr(np.load(matrix_path2))
    corr_matrix_spearman,_,_ = cluster_corr(np.load(matrix_path3))
    fig, ax = plt.subplots(ncols=3, figsize=(12, 3))
    ax[0].imshow(corr_matrix_state, aspect='auto', cmap='gray', interpolation='nearest', alpha=0.5, origin='lower')
    ax[1].imshow(corr_matrix_pearson, aspect='auto', cmap='gray', interpolation='nearest', alpha=0.5, origin='lower')
    ax[2].imshow(corr_matrix_spearman, aspect='auto', cmap='gray', interpolation='nearest', alpha=0.5, origin='lower')
    plt.savefig('matrix.png')

plot_one()