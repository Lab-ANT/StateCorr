import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, roc_curve
import seaborn as sns

use_data = 'dataset2'

def draw_boderline(X):
    for i in range(10):
        X[i*10,:]=0
        X[:,i*10]=0
    X[99,:] = 0
    X[:,99] = 0
    return X

def expand_matrix(X):
    matrix = np.ones((100, 100))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            matrix[i*10:i*10+10,j*10:j*10+10]=X[i,j]
    return matrix

script_path = os.path.dirname(__file__)
output_path = os.path.join(script_path, '../output/dataset2')
matrix_path = os.path.join(output_path, 'matrix/matrix.npy')
true_matrix_path = os.path.join(output_path, 'matrix/true_matrix.npy')
lag_matrix_path = os.path.join(output_path, 'matrix/lag_matrix.npy')
figure_output_path = os.path.join(script_path, '../output/figs')

if not os.path.exists(figure_output_path):
    os.makedirs(figure_output_path)

matrix = np.load(matrix_path)
true_matrix = expand_matrix(np.load(true_matrix_path))
print(matrix.shape, true_matrix.shape)

plt.imshow(draw_boderline(matrix))
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(figure_output_path,'matrix.png'))
plt.close()

plt.imshow(draw_boderline(true_matrix))
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(figure_output_path,'groundtruth.png'))
plt.close()

mse = np.mean(np.sqrt(np.square(matrix-true_matrix)))
print(mse)

# print(f1_score(groundtruth_matrix.flatten(), prediction.flatten()))
# print(precision_score(groundtruth_matrix.flatten(), prediction.flatten()))
# print(recall_score(groundtruth_matrix.flatten(), prediction.flatten()))

# fpr, tpr, thread = roc_curve(groundtruth_matrix.flatten(), prediction.flatten())
# plt.plot(fpr, tpr, color = 'darkorange')
# plt.savefig('roc.png')
# plt.close()