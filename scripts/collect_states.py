# Created by Chengyu on 2023/3/8.
import numpy as np
import os
from TSpy.label import seg_to_label, reorder_label, compact
from scipy import io
import matplotlib.pyplot as plt

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
        # print(data.shape)
    return data_list

def load_ActRecTut(use_data):
    dir_name = use_data
    dataset_path = os.path.join(data_path,'ActRecTut/'+dir_name+'/data.mat')
    data = io.loadmat(dataset_path)
    groundtruth = data['labels'].flatten()
    groundtruth = reorder_label(groundtruth)
    data = data['data'][:,0:10]
    return data, groundtruth

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

data = load_ActRecTut_as_classification_dataset(None)

num_states = len(data)
fig, ax = plt.subplots(nrows=num_states)
for i in range(num_states):
    ax[i].plot(data[i])
    print(data[i].shape)
plt.savefig('class.png')
# print(data[0].shape)

# from TSpy.view import plot_mts
# data, groundtruth = load_ActRecTut()
# plot_mts(data, groundtruth)
# plt.savefig('class2.png')

# data, groundtruth = load_ActRecTut_as_classification_dataset()
# data = load_USC_HAD_as_classification_dataset(1,1,data_path)

# import matplotlib.pyplot as plt
# from TSpy.view import plot_mts
# plot_mts(data,groundtruth=groundtruth)
# plt.savefig('class.png')