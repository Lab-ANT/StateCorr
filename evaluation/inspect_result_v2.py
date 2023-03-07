import numpy as np
import matplotlib.pyplot as plt
import os
from TSpy.corr import state_correlation

script_path = os.path.dirname(__file__)
state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq2/')

state_seq_list = []
for i in range(10):
    state_seq = np.load(state_seq_path+'group'+str(i)+'.npy')
    state_seq_list.append(state_seq)

matrix = state_correlation(state_seq_list)

fig, ax = plt.subplots(ncols=2, figsize=(8,3))
ax[0].imshow(matrix)
# ax[1].imshow(groundtruth_matrix)
# # plt.colorbar()
plt.savefig('lagmatrix.png')