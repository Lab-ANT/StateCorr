import numpy as np
import os
from TSpy.label import compact

use_data = 'dataset'

script_path = os.path.dirname(__file__)
true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq_'+use_data)

# def num_segs(X):
#     return len(compact(X))

def num_segs(X):
    pre = X[0]
    num_segs = 0
    pre_cut_pos = 0
    seg_length_list = []
    for i, e in enumerate(X):
        if e != pre:
            num_segs+=1
            pre = e
            seg_length = i-pre_cut_pos
            pre_cut_pos = i
            seg_length_list.append(seg_length)
    return num_segs, seg_length_list

def num_states(X):
    return len(set(X))

state_seq_list = []
state_num_list = []
for i in range(1,6):
    group_path = true_state_seq_path+str(i)+'/'
    file_list = os.listdir(group_path)
    for file_name in file_list:
        state_seq = np.load(os.path.join(group_path,file_name))
        state_seq_list.append(state_seq)

num_seg_list = []
num_state_list = []
all_seg_len_list = []
for state_seq in state_seq_list:
    num_seg, seg_len_list = num_segs(state_seq)
    num_seg_list.append(num_seg)
    all_seg_len_list+=seg_len_list
    num_state_list.append(num_states(state_seq))

print('Num of Segs: ',np.min(num_seg_list), '~', np.max(num_seg_list), np.mean(num_seg_list))
print('Num of States: ',np.min(num_state_list), '~', np.max(num_state_list), np.mean(num_state_list))
print('Seg len: ',np.min(all_seg_len_list), '~', np.max(all_seg_len_list), np.mean(all_seg_len_list))
