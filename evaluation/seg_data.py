import sys
import os
import numpy as np
import tqdm
sys.path.append('./')
sys.path.append('./Baselines/TICC')
from Baselines.TICC import *
from TICC_solver import TICC
from Time2State.time2state import Time2State
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *
from TSpy.utils import *

# from Baselines.hdphsmm import *

script_path = os.path.dirname(__file__)

def use_StateCorr(use_data):
    data_path = os.path.join(script_path, '../data/synthetic_data/'+use_data)
    file_list = os.listdir(data_path)
    output_path = os.path.join(script_path, '../output/output_StateCorr/'+use_data+'/state_seq/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    win_size = 256
    step = 50

    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['out_channels'] = 2
    params_LSE['nb_steps'] = 20
    params_LSE['compared_length'] = win_size
    params_LSE['kernel_size'] = 3

    for file_name in tqdm.tqdm(file_list):
        data = np.load(os.path.join(data_path, file_name))
        # data = normalize(data)
        params_LSE['in_channels'] = data.shape[1]
        t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
        np.save(os.path.join(script_path, '../output/output_StateCorr/'+use_data+'/state_seq/'+file_name[:-4]+'.npy'), t2s.state_seq)

def use_TICC(use_data):
    data_path = os.path.join(script_path, '../data/synthetic_data/'+use_data)
    true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq_'+use_data+'/')
    file_list = os.listdir(data_path)
    output_path = os.path.join(script_path, '../output/output_TICC/'+use_data+'/state_seq/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in tqdm.tqdm(file_list):
        data = np.load(os.path.join(data_path, file_name))
        state_seq = np.load(os.path.join(true_state_seq_path, file_name))
        win_size=3
        num_state=len(set(state_seq))
        beta=2500
        lambda_parameter=1e-3
        threshold=1e-3, 
        ticc = TICC(window_size=win_size, number_of_clusters=num_state, lambda_parameter=lambda_parameter, beta=beta, maxIters=10, threshold=threshold,
                    write_out_file=False, prefix_string="output_folder/", num_proc=10)
        prediction, _ = ticc.fit_transform(data)
        prediction = prediction.astype(int)
        np.save(output_path+file_name[:-4]+'.npy', prediction)

def use_HDPHSMM(use_data):
    data_path = os.path.join(script_path, '../data/synthetic_data/'+use_data)
    file_list = os.listdir(data_path)
    output_path = os.path.join(script_path, '../output/output_HDP-HSMM/'+use_data+'/state_seq/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in tqdm.tqdm(file_list):
        data = np.load(os.path.join(data_path, file_name))
        prediction = HDP_HSMM(1e4, 20, n_iter=20).fit(data)
        np.save(os.path.join(script_path, '../output/output_HDP-HSMM/'+use_data+'/state_seq/'+file_name[:-4]+'.npy'), prediction)


for i in range(1,6):
    dataset_name = 'dataset'+str(i)
    # use_HDPHSMM(dataset_name)
    # use_StateCorr(dataset_name)
    use_TICC(dataset_name)