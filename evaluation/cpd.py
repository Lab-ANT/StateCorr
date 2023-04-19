import sys
import os
import numpy as np
import tqdm
sys.path.append('./')
from Time2State.time2state import Time2State
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *
from TSpy.utils import *
from TSpy.label import seg_to_label, reorder_label, compact
from scipy import io

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/')

def load_ActRecTut(use_data):
    dir_name = use_data
    dataset_path = os.path.join(data_path,'ActRecTut/'+dir_name+'/data.mat')
    data = io.loadmat(dataset_path)
    groundtruth = data['labels'].flatten()
    groundtruth = reorder_label(groundtruth)
    data = data['data'][:,0:10]
    return data, groundtruth

# if not os.path.exists(output_path):
#     os.makedirs(output_path)

win_size = 256
step = 50

params_LSE['M'] = 10
params_LSE['N'] = 4
params_LSE['out_channels'] = 2
params_LSE['nb_steps'] = 20
params_LSE['compared_length'] = win_size
params_LSE['kernel_size'] = 3

data, groundtruth = load_ActRecTut('subject1_walk')
params_LSE['in_channels'] = data.shape[1]
t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
t2s.calculate_velocity()
velocity = t2s.velocity_list

    # data = np.load(os.path.join(data_path, file_name))
    # # data = normalize(data)
    # np.save(os.path.join(script_path, '../output/output_StateCorr/'+use_data+'/state_seq/'+file_name[:-4]+'.npy'), t2s.state_seq)

# def use_StateCorr(use_data):
#     data_path = os.path.join(script_path, '../data/synthetic_data/'+use_data)
#     file_list = os.listdir(data_path)
#     output_path = os.path.join(script_path, '../output/output_StateCorr/'+use_data+'/state_seq/')

#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     win_size = 256
#     step = 50

#     params_LSE['M'] = 10
#     params_LSE['N'] = 4
#     params_LSE['out_channels'] = 2
#     params_LSE['nb_steps'] = 20
#     params_LSE['compared_length'] = win_size
#     params_LSE['kernel_size'] = 3

#     for file_name in tqdm.tqdm(file_list):
#         data = np.load(os.path.join(data_path, file_name))
#         # data = normalize(data)
#         params_LSE['in_channels'] = data.shape[1]
#         t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
#         np.save(os.path.join(script_path, '../output/output_StateCorr/'+use_data+'/state_seq/'+file_name[:-4]+'.npy'), t2s.state_seq)