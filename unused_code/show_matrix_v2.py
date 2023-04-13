import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, roc_curve
from TSpy.corr import cluster_corr

use_data = 'dataset5'

script_path = os.path.dirname(__file__)
output_path = os.path.join(script_path, '../output/'+use_data+'/')
matrix_path = os.path.join(output_path, 'matrix/matrix.npy')
true_matrix_path = os.path.join(output_path, 'matrix/true_matrix.npy')
lag_matrix_path = os.path.join(output_path, 'matrix/lag_matrix.npy')
true_lag_matrix_path = os.path.join(script_path, '../data/synthetic_data/lag_matrix_'+use_data+'.npy')
figure_output_path = os.path.join(script_path, '../output/figs')

if not os.path.exists(figure_output_path):
    os.makedirs(figure_output_path)

def draw_boderline(X):
    for i in range(10):
        X[i*10,:]=0
        X[:,i*10]=0
    X[99,:] = 0
    X[:,99] = 0
    return X

matrix = np.load(matrix_path)
true_matrix = np.load(true_matrix_path)
lag_matrix = np.load(lag_matrix_path)
print(true_lag_matrix_path)
true_lag_matrix = np.load(true_lag_matrix_path)
print(matrix.shape, true_matrix.shape, lag_matrix.shape)

# plt.imshow(draw_boderline(matrix))
plt.imshow(matrix)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(figure_output_path,'matrix.png'))
plt.close()

# plt.imshow(draw_boderline(true_matrix))
plt.imshow(true_matrix, vmin=0, vmax=1)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(figure_output_path,'groundtruth.png'))
plt.close()

plt.imshow(-lag_matrix)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(figure_output_path,'lag.png'))
plt.close()

plt.imshow(true_lag_matrix)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(figure_output_path,'true_lag.png'))
plt.close()

mse = np.mean(np.sqrt(np.square(matrix-true_matrix)))
print(mse)

# from sklearn.metrics import f1_score, precision_score, recall_score
# print(f1_score((true_matrix>0.8).flatten(), (matrix>0.5).flatten()))
# print(precision_score(true_matrix.flatten(), lag_matrix.flatten()))
# print(recall_score(true_matrix.flatten(), lag_matrix.flatten()))

fpr, tpr, thread = roc_curve((true_matrix>0.8).flatten(), matrix.flatten())
plt.plot(fpr, tpr, color = 'darkorange')
plt.savefig(os.path.join(figure_output_path,'roc.png'))
print(fpr, tpr, thread)
plt.close()