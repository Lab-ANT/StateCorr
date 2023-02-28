import os
import pandas as pd
import numpy as np
from TSpy.dataset import load_SMD
import tqdm

metric = 'pearson'
# metric = 'spearman'

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/SMD/')
file_list = os.listdir(os.path.join(data_path, 'train'))

matrix_save_path = os.path.join(script_path, '../case_study1/'+metric)
if not os.path.exists(matrix_save_path):
    os.makedirs(matrix_save_path)

for file_name in tqdm.tqdm(file_list):
    file_name = file_name[:-4]
    _,_,_,_,data,_ = load_SMD(data_path, file_name)
    corr_matrix = pd.DataFrame(data).corr(metric).to_numpy()
    np.save(os.path.join(matrix_save_path, file_name+'.npy'), corr_matrix)