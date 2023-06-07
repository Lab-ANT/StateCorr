import sys
import os
import numpy as np
import tqdm
sys.path.append('./')
sys.path.append('./Baselines/TICC')
from Baselines.TICC import *
from TICC_solver import TICC
from TSpy.utils import *

script_path = os.path.dirname(__file__)

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

# use_TICC('dataset1')

for i in range(1,6):
    dataset_name = 'dataset'+str(i)
    # use_HDPHSMM(dataset_name)
    # use_StateCorr(dataset_name)
    use_TICC(dataset_name)