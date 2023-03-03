import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from TSpy.corr import state_correlation, lagged_state_correlation

metric = ''
# metric = '128'
# metric = '256'
# metric = '512'
# metric = '1024'

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/SMD/')
file_list = os.listdir(os.path.join(data_path, 'train'))

matrix_save_path = os.path.join(script_path, '../case_study1/state'+metric)
if not os.path.exists(matrix_save_path):
    os.makedirs(matrix_save_path)

def calculate_all():
    for file_name in tqdm.tqdm(file_list):
        file_name = file_name[:-4]
        result_path = os.path.join(script_path, '../case_study1/state_seq-'+metric+'/'+file_name+'.npy')
        state_seq_array = np.load(result_path)
        corr_matrix = state_correlation(state_seq_array)
        # corr_matrix = lagged_state_correlation(state_seq_array)
        np.save(os.path.join(matrix_save_path, file_name+'.npy'), corr_matrix)

def calculate_one(file_name):
    result_path = os.path.join(script_path, '../case_study1/state_seq'+metric+'/'+file_name+'.npy')
    state_seq_array = np.load(result_path)
    # corr_matrix = state_correlation(state_seq_array)
    corr_matrix = lagged_state_correlation(state_seq_array)
    np.save(os.path.join(matrix_save_path, file_name+'.npy'), corr_matrix)

# calculate_all()
calculate_one('machine-1-6')