# Created by Chengyu on 2023/3/8.
import numpy as np
import os
from TSpy.label import seg_to_label, reorder_label, compact
from scipy import io
import matplotlib.pyplot as plt
import pandas as pd
from TSpy.utils import *

def fill_nan(data):
    x_len, y_len = data.shape
    for x in range(x_len):
        for y in range(y_len):
            if np.isnan(data[x,y]):
                data[x,y]=data[x-1,y]
    return data

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/')
save_path = os.path.join(script_path, '../data/synthetic_data/')

def load_USC_HAD_as_classification_dataset(subject, target, dataset_path):
    prefix = os.path.join(dataset_path,'USC-HAD/Subject'+str(subject)+'/')
    fname_prefix = 'a'
    fname_postfix = 't'+str(target)+'.mat'
    data_list = []
    for i in range(1,13):
        data = io.loadmat(prefix+fname_prefix+str(i)+fname_postfix)
        data = data['sensor_readings']
        data_list.append(data)
    return data_list

def load_ActRecTut(use_data):
    dir_name = use_data
    dataset_path = os.path.join(data_path,'ActRecTut/'+dir_name+'/data.mat')
    data = io.loadmat(dataset_path)
    groundtruth = data['labels'].flatten()
    groundtruth = reorder_label(groundtruth)
    data = data['data'][:,0:10]
    return data, groundtruth

def load_PAMAP2():
    dataset_path = os.path.join(data_path,'PAMAP2/Protocol/subject10'+str(1)+'.dat')
    df = pd.read_csv(dataset_path, sep=' ', header=None)
    data = df.to_numpy()
    groundtruth = np.array(data[:,1],dtype=int)
    hand_acc = data[:,4:7]
    chest_acc = data[:,21:24]
    ankle_acc = data[:,38:41]
    data = np.hstack([hand_acc, chest_acc, ankle_acc])
    data = fill_nan(data)
    # data = normalize(data)
    return data, groundtruth

def load_PAMAP2_as_classification_dataset(use_state=None):
    data, groundtruth = load_PAMAP2()
    groundtruth = reorder_label(groundtruth)
    print(data.shape)
    num_states = len(set(groundtruth))

    if use_state is None:
        use_state = [int(i) for i in range(num_states)]

    data_list = []
    for state in use_state:
        idx = np.argwhere(groundtruth==state)
        print(idx)
        data_list.append(data[idx].squeeze(1))
    return data_list

def load_ActRecTut_as_classification_dataset(use_state=None):
    data1, groundtruth1 = load_ActRecTut('subject1_walk')
    data2, groundtruth2 = load_ActRecTut('subject2_walk')

    data = np.concatenate([data1, data2])
    groundtruth = np.concatenate([groundtruth1, groundtruth2])
    print(data.shape, groundtruth.shape)

    num_states = len(set(groundtruth))
    
    if use_state is None:
        use_state = [i for i in range(num_states)]

    data_list = []
    for state in use_state:
        idx = np.argwhere(groundtruth==state)
        data_list.append(data[idx].squeeze(1))
    return data_list

class_list = []
class_list += load_PAMAP2_as_classification_dataset(None)
# class_list += load_ActRecTut_as_classification_dataset(None)
# class_list += load_USC_HAD_as_classification_dataset(1, 1, data_path)
print(len(class_list))

for c in class_list:
    print(len(c))
# print(len(class_list))

# num_states = len(data)
# fig, ax = plt.subplots(nrows=num_states)
# for i in range(num_states):
#     ax[i].plot(data[i])
#     print(data[i].shape)
# plt.savefig('class.png')