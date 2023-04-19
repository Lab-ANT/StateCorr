import os
import pandas as pd
import matplotlib.pyplot as plt
from TSpy.label import seg_to_label
from TSpy.utils import len_of_file
import numpy as np

script_path = os.path.dirname(__file__)
result_save_path = os.path.join(script_path, '../segment_results')
data_path = os.path.join(script_path, '../data/')
result_save_path = os.path.join(script_path, '../segment_results')
fig_save_path = os.path.join(script_path, 'figs')

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_path(fig_save_path)

def plot_results(method, dataset):
    dataset_path = os.path.join(data_path,'UCR-SEG/')
    for fname in os.listdir(dataset_path):
        info_list = fname[:-4].split('_')
        f = info_list[0]
        print(f)
        window_size = int(info_list[1])
        seg_info = {}
        i = 0
        for seg in info_list[2:]:
            seg_info[int(seg)]=i
            i+=1
        seg_info[len_of_file(dataset_path+fname)]=i
        df = pd.read_csv(dataset_path+fname)
        data = df.to_numpy()
        groundtruth = seg_to_label(seg_info)[:-1]
        plt.style.use('classic')
        fig, ax = plt.subplots(nrows=4)
        ax[0].plot(data)
        ax[0].set_xlim([0, len(data)])
        prediction = np.load(os.path.join(result_save_path, 'UCR-SEG/Time2State/'+fname+'.npy'))
        ax[1].imshow(groundtruth.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')
        ax[2].imshow(prediction.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')
        plt.savefig(os.path.join(fig_save_path, fname+'.png'))
        plt.close()

plot_results(None, None)