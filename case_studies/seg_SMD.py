import sys
import os
import numpy as np
import tqdm
from TSpy.dataset import load_SMD

sys.path.append('./')
from Time2State.time2state import Time2State
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/SMD/')
train_path = os.path.join(data_path, 'train')
file_list = os.listdir(os.path.join(data_path, 'train'))

# 1024 is about one-day-long
win_size =256
step = 50

params_LSE['in_channels'] = 1
params_LSE['out_channels'] = 1
params_LSE['nb_steps'] = 20
params_LSE['compared_length'] = win_size

if not os.path.exists('case_study1/state_seq/'):
    os.makedirs('case_study1/state_seq/')

def seg_all():
    for file_name in file_list:
        file_name = file_name[:-4]
        _,_,_,_,data,_ = load_SMD(data_path, file_name)
        state_seq_array = []
        for i in range(data.shape[1]):
            t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data[:,i].reshape(-1,1), win_size, step)
            state_seq_array.append(t2s.state_seq)
        state_seq_array = np.array(state_seq_array)
        np.save('case_study1/state_seq/'+file_name+'.npy', state_seq_array)
        print(file_name)

def seg_one(file_name):
    data,_,_,_,_,_ = load_SMD(data_path, file_name)
    data = data[::4]
    state_seq_array = []
    for i in tqdm.tqdm(range(data.shape[1])):
        t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data[:,i].reshape(-1,1), win_size, step)
        state_seq_array.append(t2s.state_seq)
    state_seq_array = np.array(state_seq_array)
    np.save('case_study1/state_seq/'+file_name+'.npy', state_seq_array)
    print(file_name)

# seg_all()
seg_one('machine-1-6')