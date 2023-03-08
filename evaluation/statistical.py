import numpy as np
import os

use_data = 'dataset3'

script_path = os.path.dirname(__file__)
true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq_'+use_data+'/')

state_seq_list = []
state_num_list = []
for i in range(10):
    state_seq = np.load(true_state_seq_path+'group'+str(i)+'.npy')
    state_num_list.append(len(set(state_seq)))

print(state_num_list)
