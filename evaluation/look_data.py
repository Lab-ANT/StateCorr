import os
import numpy as np
import tqdm
from TSpy.view import plot_mts
import matplotlib.pyplot as plt

script_path = os.path.dirname(__file__)

def use_StateCorr(use_data):
    data_path = os.path.join(script_path, '../data/synthetic_data/'+use_data)
    file_list = os.listdir(data_path)
    output_path = os.path.join(script_path, '../output/output_StateCorr/'+use_data+'/state_seq/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in tqdm.tqdm(file_list):
        data = np.load(os.path.join(data_path, file_name))
        plot_mts(data)
        plt.savefig('fig.png')
        plt.close()
        # np.save(os.path.join(script_path, '../output/output_StateCorr/'+use_data+'/state_seq/'+file_name[:-4]+'.npy'), t2s.state_seq)

# for i in range(1,6):
#     dataset_name = 'dataset'+str(i)
use_StateCorr('dataset1')