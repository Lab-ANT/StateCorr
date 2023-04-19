from cProfile import label
import sys
import os
sys.path.append('./')
from Time2State.time2state import Time2State
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *
from TSpy.utils import *
from TSpy.label import seg_to_label, reorder_label, compact
from scipy import io
import matplotlib.pyplot as plt

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/')

def load_ActRecTut(use_data):
    dir_name = use_data
    dataset_path = os.path.join(data_path,'ActRecTut/'+dir_name+'/data.mat')
    data = io.loadmat(dataset_path)
    groundtruth = data['labels'].flatten()
    groundtruth = reorder_label(groundtruth)
    data = data['data'][:,0:10]
    return data, groundtruth

win_size = 256
step = 50

params_LSE['M'] = 10
params_LSE['N'] = 4
params_LSE['out_channels'] = 2
params_LSE['nb_steps'] = 20
params_LSE['compared_length'] = win_size
params_LSE['kernel_size'] = 3

data, groundtruth = load_ActRecTut('subject1_walk')
params_LSE['in_channels'] = data.shape[1]
t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)

plt.style.use('classic')
fig, ax = plt.subplots(nrows=4)
ax[0].plot(data)
ax[0].set_xlim([0, len(data)])
ax[1].plot(t2s.velocity)
ax[1].set_xlim([0, len(t2s.velocity)])
ax[1].hlines(np.mean(t2s.velocity), 0, len(t2s.velocity), color="red")
ax[2].imshow(groundtruth.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')
ax[3].imshow(t2s.state_seq.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')

# max_v = np.max(velocity)
# min_v = np.min(velocity)
# for cut in cut_list:
#     ax[1].vlines(cut, min_v, max_v,color="red")

plt.savefig('velocity.png')