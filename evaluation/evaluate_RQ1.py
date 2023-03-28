import numpy as np
import os
import matplotlib.pyplot as plt
from TSpy.corr import lagged_state_correlation
from TSpy.label import reorder_label

num_in_group = 20
method_list = ['StateCorr']
for method in method_list:
    prediction_list = []
    gt_list = []
    for i in range(1,3):
        use_data = 'dataset'+str(i)
        script_path = os.path.dirname(__file__)
        data_path = os.path.join(script_path, '../output/output_'+method+'/'+use_data+'/')
        true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq_'+use_data+'/')
        figure_output_path = os.path.join(script_path, '../output/output_'+method+'/figs')

        for j in range(num_in_group):
            # prediction may be shorter than groundtruth
            # this is caused by the sliding window and the difference is very short
            # we simply cut the groundtruth
            prediction = reorder_label(np.load(os.path.join(data_path, 'state_seq/test'+str(j)+'.npy')))
            groundtruth = reorder_label(np.load(true_state_seq_path+'test'+str(j)+'.npy'))[:len(prediction)]
            print(j, prediction.shape, groundtruth.shape)
            gt_list.append(groundtruth)
            prediction_list.append(prediction)
    
    # calculate correlation matrix and lag matrix
    # matrix, lag_matrix = lagged_state_correlation(prediction_list)
    # print(matrix.shape, lag_matrix.shape)

    groundtruth_matrix = np.zeros((3*num_in_group,3*num_in_group))
    plt.imshow(groundtruth_matrix)
    plt.savefig(os.path.join(figure_output_path, 'groundtruth_matrix.png'))
