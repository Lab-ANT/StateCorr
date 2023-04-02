import os
import numpy as np
import tqdm
from TSpy.view import plot_mts
import matplotlib.pyplot as plt
from TSpy.label import reorder_label

script_path = os.path.dirname(__file__)

def use_StateCorr(use_data):
    data_path = os.path.join(script_path, '../data/synthetic_data/'+use_data)
    file_list = os.listdir(data_path)
    output_path = os.path.join(script_path, '../output/output_StateCorr/'+use_data+'/state_seq/')
    true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq_'+use_data+'/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in tqdm.tqdm(file_list):
        data = np.load(os.path.join(data_path, file_name))
        groundtruth = reorder_label(np.load(true_state_seq_path+file_name)[:len(data)])
        plot_mts(data, groundtruth)
        plt.savefig('fig.png')
        plt.close()
        # np.save(os.path.join(script_path, '../output/output_StateCorr/'+use_data+'/state_seq/'+file_name[:-4]+'.npy'), t2s.state_seq)

# for i in range(1,6):
#     dataset_name = 'dataset'+str(i)
use_StateCorr('dataset1')