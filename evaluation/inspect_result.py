import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from TSpy.corr import state_correlation

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../output/effectiveness_of_StateCorr/')

state_seq_array = []
for i in range(100):
    state_seq = np.load(os.path.join(data_path, 'test'+str(i)+'.npy'))
    state_seq_array.append(state_seq)

state_seq_array = np.array(state_seq_array)
matrix = state_correlation(state_seq_array)

plt.imshow(matrix)
plt.colorbar()
plt.savefig('corr_matrix.png')