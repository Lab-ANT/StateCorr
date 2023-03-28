import numpy as np
import os
import sys
import tqdm

sys.path.append('./')
from Time2State.time2state import Time2State
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *

use_data = 'dataset5'

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/synthetic_data/'+use_data)
file_list = os.listdir(data_path)
output_path = os.path.join(script_path, '../output/'+use_data+'/state_seq/')

if not os.path.exists(output_path):
    os.makedirs(output_path)

win_size = 256
step = 50

params_LSE['in_channels'] = 4
params_LSE['M'] = 10
params_LSE['N'] = 4
params_LSE['out_channels'] = 2
params_LSE['nb_steps'] = 10
params_LSE['compared_length'] = win_size
params_LSE['kernel_size'] = 3

for file_name in tqdm.tqdm(file_list):
    data = np.load(os.path.join(data_path, file_name))
    t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
    np.save(os.path.join(script_path, '../output/'+use_data+'/state_seq/'+file_name[:-4]+'.npy'), t2s.state_seq)