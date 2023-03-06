import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, roc_curve

script_path = os.path.dirname(__file__)
groundtruth_matrix_path = os.path.join(script_path, '../data/synthetic_data/groundtruth_matrix_d1.npy')
groundtruth_matrix = np.load(groundtruth_matrix_path)==1

matrix = np.load(os.path.join(script_path, '../output/matrix/matrix.npy'))
lag_matrix = np.load(os.path.join(script_path, '../output/matrix/lag_matrix.npy'))

prediction = matrix
# print(f1_score(groundtruth_matrix.flatten(), prediction.flatten()))
# print(precision_score(groundtruth_matrix.flatten(), prediction.flatten()))
# print(recall_score(groundtruth_matrix.flatten(), prediction.flatten()))

fpr, tpr, thread = roc_curve(groundtruth_matrix.flatten(), prediction.flatten())
plt.plot(fpr, tpr, color = 'darkorange')
plt.savefig('roc.png')
plt.close()

fig, ax = plt.subplots(ncols=2, figsize=(8,3))
ax[0].imshow(matrix)
ax[1].imshow(lag_matrix)
# plt.colorbar()
plt.savefig('lagmatrix.png')