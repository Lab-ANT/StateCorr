import sys
import os
import numpy as np
sys.path.append('./')
from TSpy.utils import *
from TSpy.view import plot_mts

script_path = os.path.dirname(__file__)

def use_StateCorr(use_data, num):
    data_path = os.path.join(script_path, '../data/synthetic_data/'+use_data)
    file_list = os.listdir(data_path)

    data = np.load(os.path.join(data_path, 'test%d.npy'%(num)))
    groundtruth = np.load(os.path.join(script_path, '../data/synthetic_data/state_seq_%s/test%d.npy'%(use_data, num)))
    state_seq = np.load(os.path.join(script_path, '../output/output_StateCorr/'+use_data+'/state_seq/test%d.npy'%(num)))

    plot_mts(data, groundtruth, state_seq)
    import matplotlib.pyplot as plt
    plt.savefig('result.png')

    print(data.shape, groundtruth.shape, state_seq.shape)

use_StateCorr('dataset5', 1)