import sys
import os
import numpy as np
import tqdm
sys.path.append('./')
# sys.path.append('./Baselines/TICC')
# from Baselines.TICC import *
# from TICC_solver import TICC
# from Time2State.time2state import Time2State
# from Time2State.adapers import *
# from Time2State.clustering import *
# from Time2State.default_params import *
# from TSpy.utils import *
from Baselines.hdphsmm import *

script_path = os.path.dirname(__file__)

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
    use_HDPHSMM(dataset_name)
    # use_StateCorr(dataset_name)
    # use_TICC(dataset_name)