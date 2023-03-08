# Created by Chengyu on 2023/3/8.
import numpy as np
import pandas as pd
import os
from shape import RMDF
from TSpy.label import seg_to_label

# configuration
channel_num = 4
# seg_num = 20
seg_len = [500, 1000] # 200~1000
state_num = [4, 8] # 4~8
num_group = 10
num_ts_in_group = 10
script_path = os.path.dirname(__file__)
save_path = os.path.join(script_path, '../data/synthetic_data/')
dataset_name = 'dataset3'
random_state = None

# Generate segment json
length = 20000

def generate_seg_json(seg_len, state_num, random_state=None):
    # Config seed to generate determinally.
    if random_state is not None:
        np.random.seed(random_state)
    seg_json = {}
    # maximum possible num of segments.
    seg_num = int(length/seg_len[0])
    # generate random state num.
    random_state_num = np.random.randint(low=state_num[0], high=state_num[1]+1)
    # generate state for each segment.
    state_list = np.random.randint(random_state_num, size=seg_num)
    while len(set(state_list)) != random_state_num:
        state_list = np.random.randint(random_state_num, size=seg_num)
    # generate length for each segment.
    seg_len_list = np.random.randint(low=seg_len[0], high=seg_len[1], size=seg_num)
    total_len = 0
    for i, state, rand_seg_len in zip(range(seg_num), state_list, seg_len_list):
        total_len += rand_seg_len
        if total_len>=length:
            total_len = length
        seg_json[total_len]=state
    return seg_json

def gen_channel_from_json(seg_json):
    state_list = [seg_json[seg] for seg in seg_json]
    seg_len_list = np.array([seg for seg in seg_json])
    first_seg_len = seg_len_list[0]
    seg_len_list = np.insert(np.diff(seg_len_list), 0, first_seg_len)
    true_state_num = len(set(state_list))
    # This is an object list.
    rmdf_list = [RMDF.RMDF(depth=5) for i in range(true_state_num)]
    for rmdf in rmdf_list:
        rmdf.gen_anchor()
    seg_list = []
    for state, seg_len in zip(state_list, seg_len_list):
        seg = [rmdf_list[state].gen(forking_depth=1, length=100) for i in range(10)]
        seg_list.append(np.concatenate(seg)[:seg_len])
    # print(true_state_num)
    result = np.concatenate(seg_list)
    return result

def gen_from_json(seg_json):
    # generate channel respectively.
    channel_list = [gen_channel_from_json(seg_json) for i in range(channel_num)]
    return np.stack(channel_list).T

def generate_group(num_ts_in_group, seg_json):
    data_list = []
    for i in range(num_ts_in_group):
        data = np.concatenate([gen_from_json(seg_json)])
        data_list.append(data)
    return data_list

seg_json_list = []
for i in range(num_group):
    seg_json =generate_seg_json(seg_len, state_num, random_state)
    seg_json_list.append(seg_json)

group_list = [generate_group(num_ts_in_group, seg_json) for seg_json in seg_json_list]

width = num_ts_in_group * num_group
groundtruth_matrix = np.zeros(shape=(width, width))
for i in range(num_group):
    start = i*num_ts_in_group
    end = (i+1)*num_ts_in_group
    groundtruth_matrix[start:end, start:end] = 1

full_path = save_path+dataset_name
if not os.path.exists(full_path):
    os.makedirs(full_path)
if not os.path.exists(save_path+'state_seq_'+dataset_name):
    os.makedirs(save_path+'state_seq_'+dataset_name)

i = 0
for group in group_list:
    for data in group:
        df = pd.DataFrame(data).round(4)
        df.to_csv(full_path+'/test'+str(i)+'.csv', header=False)
        i += 1

np.save(save_path+'matrix_'+dataset_name+'.npy', groundtruth_matrix)

state_seq_list = [seg_to_label(seg_json) for seg_json in seg_json_list]
for i, state_seq in enumerate(state_seq_list):
    np.save(save_path+'state_seq_'+dataset_name+'/group'+str(i)+'.npy', state_seq)

# print(np.random.randint(low=4, high=8+1, size=100))