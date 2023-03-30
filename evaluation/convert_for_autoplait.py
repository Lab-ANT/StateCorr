import os
import numpy as np
import tqdm
import pandas as pd

script_path = os.path.dirname(__file__)
data_save_path = os.path.join(script_path,'../data/synthetic_data_AutoPlait')
data_original_path = os.path.join(script_path, '../data/synthetic_data/')

if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

for i in tqdm.tqdm(range(1,6)):
    if not os.path.exists(os.path.join(data_save_path, 'dataset%d'%(i))):
        os.makedirs(os.path.join(data_save_path, 'dataset%d'%(i)))
    for j in range(20):
        data = np.load(os.path.join(data_original_path, 'dataset%d/test%d.npy'%(i,j)))
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(data_save_path, 'dataset%d/test%d.csv'%(i,j)), header=None, index=None, sep=' ')