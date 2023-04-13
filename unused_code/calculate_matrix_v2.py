import numpy as np
import os
from TSpy.corr import state_correlation, lagged_state_correlation
import matplotlib.pyplot as plt

use_data = 'dataset2'

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../output/'+use_data+'/')
true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq_'+use_data+'/')

if not os.path.exists(data_path+'matrix'):
    os.makedirs(data_path+'matrix')

state_seq_list = []

for i in range(20):
    state_seq = np.load(true_state_seq_path+'test'+str(i)+'.npy')
    state_seq_list.append(state_seq)
    plt.plot(state_seq)
plt.savefig('state.png')

true_matrix = state_correlation(state_seq_list)
np.save(os.path.join(script_path, '../output/'+use_data+'/matrix/true_matrix.npy'),true_matrix)

state_seq_array = []
for i in range(20):
    state_seq = np.load(os.path.join(data_path, 'state_seq/test'+str(i)+'.npy'))
    state_seq_array.append(state_seq)

state_seq_array = np.array(state_seq_array)
matrix, lag_matrix = lagged_state_correlation(state_seq_array)
# matrix = state_correlation(state_seq_array)
# np.save(os.path.join(script_path, '../output/'+use_data+'/matrix/matrix.npy'),matrix)
np.save(os.path.join(script_path, '../output/'+use_data+'/matrix/lag_matrix.npy'),lag_matrix)