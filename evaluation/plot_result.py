import sys
import os
import numpy as np
sys.path.append('./')
from TSpy.utils import *
from TSpy.view import plot_mts

script_path = os.path.dirname(__file__)

def use_StateCorr(use_data):
    data_path = os.path.join(script_path, '../data/synthetic_data/'+use_data)
    file_list = os.listdir(data_path)

    data = np.load(os.path.join(data_path, 'test2.npy'))
    groundtruth = np.load(os.path.join(script_path, '../data/synthetic_data/state_seq_dataset1/test2.npy'))
    state_seq = np.load(os.path.join(script_path, '../output/output_StateCorr/'+use_data+'/state_seq/test2.npy'))

    plot_mts(data, groundtruth, state_seq)
    import matplotlib.pyplot as plt
    plt.savefig('result.png')

    print(data.shape, groundtruth.shape, state_seq.shape)

use_StateCorr('dataset1')