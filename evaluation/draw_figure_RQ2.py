import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

num_in_group = 5
# method_list = ['StateCorr', 'TICC', 'AutoPlait', 'ClaSP', 'HDP-HSMM']
method_list = ['AutoPlait']

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
    
    groundtruth_matrix = np.load(os.path.join(matrix_save_path, 'groundtruth_matrix.npy'))
    prediction_matrix = np.load(os.path.join(matrix_save_path, 'prediction_matrix.npy'))
    fig, ax = plt.subplots(ncols=2, figsize=(8,4))
    ax[0].imshow(groundtruth_matrix)
    ax[1].imshow(prediction_matrix)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_output_path, 'gt_prediction_matrix.png'))
    plt.close()

    # calculate correlation matrix and lag matrix
    groundtruth_matrix = np.load(os.path.join(matrix_save_path, 'lagged_groundtruth_matrix.npy'))
    prediction_matrix =  np.load(os.path.join(matrix_save_path, 'lagged_prediction_matrix.npy'))
    lag_matrix = np.load(os.path.join(matrix_save_path, 'lag_matrix.npy'))
    # For debug, generally unused.
    # fig, ax = plt.subplots(ncols=2, figsize=(8,4))
    # ax[0].imshow(groundtruth_matrix)
    # ax[1].imshow(prediction_matrix)
    # plt.tight_layout()
    # plt.savefig(os.path.join(figure_output_path, 'gt_prediction_matrix2.png'))
    # plt.close()

    precision, recall, threshold = precision_recall_curve(groundtruth_matrix.flatten(), prediction_matrix.flatten())
    f1 = 2*precision*recall/(precision+recall)
    idx = np.argmax(f1)
    print('Best (F1-score, P, R) of '+method+': ', f1[idx], precision[idx], recall[idx])

    np.save(os.path.join(script_path, '../output/output_'+method+'/precision_RQ2.npy'), precision)
    np.save(os.path.join(script_path, '../output/output_'+method+'/recall_RQ2.npy'), recall)
    # plt.style.use('classic')
    # plt.grid()
    # plt.plot(precision, recall, lw=2)
    # plt.xlabel('Recall',fontsize=18)
    # plt.ylabel('Precision',fontsize=18)
    # plt.xticks([0.2,0.4,0.6,0.8,1.0],fontsize=14)
    # plt.yticks([0.2,0.4,0.6,0.8,1.0],fontsize=14)
    # plt.savefig(os.path.join(figure_output_path,'prc_RQ2.png'))
    # plt.close()