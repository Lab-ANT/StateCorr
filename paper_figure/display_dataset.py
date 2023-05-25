import numpy as np
import os
from TSpy.corr import lagged_state_correlation, state_correlation
from TSpy.label import reorder_label
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

num_in_group = 5

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/synthetic_data/')
dataset_name = 'dataset1'
rue_state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq_'+dataset_name+'/')

fig, ax = plt.subplots(nrows=5)
for i in range(num_in_group):
    data = np.load(os.path.join(data_path, '%s/test%d.npy'%(dataset_name, i)))
    print(data.shape)
    ax[i].plot(data)
plt.savefig('dataset.png')