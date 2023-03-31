import numpy as np
import os
from TSpy.corr import lagged_state_correlation, state_correlation
from TSpy.label import reorder_label
from sklearn.metrics import precision_recall_curve

num_in_group = 5
method_list = ['StateCorr', 'TICC', 'AutoPlait', 'ClaSP', 'HDP-HSMM']

for method in method_list:
    prediction_list = []
    gt_list = []
    for i in range(1,4):
        use_data = 'dataset'+str(i)
        script_path = os.path.dirname(__file__)
        data_path = os.path.join(script_path, '../output/output_'+method+'/'+use_data+'/')
        true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq_'+use_data+'/')
        figure_output_path = os.path.join(script_path, '../output/output_'+method+'/figs')
        matrix_save_path = os.path.join(script_path, '../output/output_'+method+'/matrix_RQ2')

        if not os.path.exists(figure_output_path):
            os.makedirs(figure_output_path)
        if not os.path.exists(matrix_save_path):
            os.makedirs(matrix_save_path)

        for j in range(num_in_group):
            # prediction may be shorter than groundtruth
            # this is caused by the sliding window and the difference is very short
            # we simply cut the groundtruth
            prediction = reorder_label(np.load(os.path.join(data_path, 'state_seq/test'+str(j)+'.npy')))
            groundtruth = reorder_label(np.load(true_state_seq_path+'test'+str(j)+'.npy'))[:len(prediction)]
            # print(j, prediction.shape, groundtruth.shape)
            gt_list.append(groundtruth)
            prediction_list.append(prediction)
    
    # calculate correlation matrix
    # without lag
    prediction_matrix = state_correlation(prediction_list)
    groundtruth_matrix = state_correlation(gt_list)
    np.save(os.path.join(matrix_save_path, 'groundtruth_matrix.npy'), prediction_matrix)
    np.save(os.path.join(matrix_save_path, 'prediction_matrix.npy'), groundtruth_matrix)

    # calculate correlation matrix and lag matrix
    groundtruth_matrix = np.zeros((3*num_in_group,3*num_in_group))
    for i in range(3):
        groundtruth_matrix[i*num_in_group:num_in_group*(i+1),i*num_in_group:num_in_group*(i+1)] = 1
    groundtruth_matrix = groundtruth_matrix==1
    prediction_matrix, lag_matrix = lagged_state_correlation(prediction_list)
    # save matrics
    np.save(os.path.join(matrix_save_path, 'lagged_groundtruth_matrix.npy'), groundtruth_matrix)
    np.save(os.path.join(matrix_save_path, 'lagged_prediction_matrix.npy'), prediction_matrix)
    np.save(os.path.join(matrix_save_path, 'lag_matrix.npy'), lag_matrix)

    precision, recall, threshold = precision_recall_curve(groundtruth_matrix.flatten(), prediction_matrix.flatten())
    f1 = 2*precision*recall/(precision+recall)
    idx = np.argmax(f1)
    print('Best (F1-score, P, R) of '+method+': ', f1[idx], precision[idx], recall[idx])