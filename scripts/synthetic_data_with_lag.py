# Created by Chengyu on 2023/3/8.
import numpy as np
import os
from shape import RMDF
import tqdm
from TSpy.label import seg_to_label

# configuration
channel_num = 4
# seg_num = 20
seg_len = [800, 1200] # 200~1000
# state_num = [4, 8] # 4~8
state_num = 8
num_group = 1
num_ts_in_group = 20
script_path = os.path.dirname(__file__)
save_path = os.path.join(script_path, '../data/synthetic_data/')
dataset_name = 'dataset5'
random_state = None
length = 20000

def gen_seg_json(state_num, seg_len):
    seg_num = int(length/seg_len[0])
    # generate state for each segment.
    state_list = np.random.randint(state_num, size=seg_num)
    # print(len(set(state_list)), state_num, state_list)
    while len(set(state_list)) != state_num:
        state_list = np.random.randint(state_num, size=seg_num)
    # generate length for each segment.
    seg_len_list = np.random.randint(low=seg_len[0], high=seg_len[1], size=seg_num)
    seg_json = {}
    total_len = 0
    for i, state, rand_seg_len in zip(range(seg_num), state_list, seg_len_list):
        total_len += rand_seg_len
        if total_len>=length:
            total_len = length
            seg_json[total_len]=state
            break
        seg_json[total_len]=state
    # print(state_num, seg_json)
    return seg_json

# Generate segment json
def generate_seg_json(seg_len, state_num, random_state=None):
    # Config seed to generate determinally.
    if random_state is not None:
        np.random.seed(random_state)
    # generate random state num.
    # random_state_num = np.random.randint(low=state_num[0], high=state_num[1]+1)
    random_state_num = state_num
    # generate state for each segment.
    seg_json = gen_seg_json(random_state_num, seg_len)
    state_list = [seg_json[seg] for seg in seg_json]
    while len(set(state_list)) != random_state_num:
        seg_json = gen_seg_json(random_state_num, seg_len)
        state_list = [seg_json[seg] for seg in seg_json]
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
        seg = [rmdf_list[state].gen(forking_depth=1, length=200) for i in range(15)]
        seg_list.append(np.concatenate(seg)[:seg_len])
    result = np.concatenate(seg_list)
    return result

def gen_from_json(seg_json):
    # generate channel respectively.
    channel_list = [gen_channel_from_json(seg_json) for i in range(channel_num)]
    return np.stack(channel_list).T

def add_lag(seg_json):
    lag = np.random.randint(low=-500, high=500)
    new_seg_json = {}
    if lag >=0:
        for seg in list(seg_json):
            if seg+lag <= length:
                new_seg_json[seg+lag] = seg_json[seg]
            else:
                new_seg_json[length] = seg_json[seg]
                break
    else:
        last_seg = list(seg_json)[-1]
        for seg in list(seg_json)[:-1]:
            new_seg_json[seg+lag] = seg_json[seg]
        new_seg_json[length] = seg_json[last_seg]
    return new_seg_json, lag

def calculate_lag_matrix(drift_array):
    lag_matrix = np.ones((num_group*num_ts_in_group, num_group*num_ts_in_group))
    for i in range(num_group*num_ts_in_group):
        for j in range(num_group*num_ts_in_group):
            if i==j:
                lag_matrix[i,j]=0
            elif i<j:
                drift1 = drift_array[i]
                drift2 = drift_array[j]
                lag = drift2-drift1
                lag_matrix[i,j]=lag
                lag_matrix[j,i]=-lag
    return lag_matrix

seg_json_list = []
for i in range(num_group):
    seg_json = generate_seg_json(seg_len, state_num, random_state)
    seg_json_list.append(seg_json)

drift_array = []
lagged_seg_json_list = []
for seg_json in seg_json_list:
    for i in range(num_ts_in_group):
        lagged_seg_json, drift = add_lag(seg_json)
        # print(lagged_seg_json, drift)
        drift_array.append(drift)
        lagged_seg_json_list.append(lagged_seg_json)

lag_matrix = calculate_lag_matrix(np.array(drift_array))

full_path = save_path+dataset_name
if not os.path.exists(full_path):
    os.makedirs(full_path)
if not os.path.exists(save_path+'state_seq_'+dataset_name):
    os.makedirs(save_path+'state_seq_'+dataset_name)

for i in tqdm.tqdm(range(num_group*num_ts_in_group)):
    data = np.concatenate([gen_from_json(lagged_seg_json_list[i])])
    state_seq = seg_to_label(lagged_seg_json_list[i])
    np.save(full_path+'/test'+str(i), data)
    np.save(save_path+'state_seq_'+dataset_name+'/test'+str(i), state_seq)
    print(state_seq.shape)
np.save(save_path+'lag_matrix_'+dataset_name, lag_matrix)