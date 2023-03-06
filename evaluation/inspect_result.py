import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from TSpy.corr import state_correlation
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, roc_curve

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../output/effectiveness_of_StateCorr/')
groundtruth_matrix_path = os.path.join(script_path, '../data/synthetic_data/groundtruth_matrix_d1.npy')

groundtruth_matrix = np.load(groundtruth_matrix_path)==1

state_seq_array = []
for i in range(100):
    state_seq = np.load(os.path.join(data_path, 'test'+str(i)+'.npy'))
    state_seq_array.append(state_seq)

state_seq_array = np.array(state_seq_array)
matrix = state_correlation(state_seq_array)

# matrix = pd.DataFrame(state_seq_array).corr('pearson').to_numpy()

prediction = matrix
# prediction = np.random.rand(10000).reshape(100,100)
# print(prediction, groundtruth_matrix)
# matrix[matrix>0.4]=1
# prediction = matrix>0.4
# print(f1_score(groundtruth_matrix.flatten(), prediction.flatten()))
# print(precision_score(groundtruth_matrix.flatten(), prediction.flatten()))
# print(recall_score(groundtruth_matrix.flatten(), prediction.flatten()))

fpr, tpr, thread = roc_curve(groundtruth_matrix.flatten(), prediction.flatten())
# fpr, tpr, thread = precision_recall_curve(groundtruth_matrix.flatten(), prediction.flatten())
print(fpr, tpr, thread)
plt.plot(fpr, tpr, color = 'darkorange')
# plt.plot(tpr, fpr, color = 'darkorange')
plt.savefig('roc.png')

# plt.imshow(matrix)
# plt.colorbar()
# plt.savefig('corr_matrix.png')
# plt.close()