import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from TSpy.label import reorder_label

num = 5
method_list = ['StateCorr', 'TICC']
script_path = os.path.dirname(__file__)

def evaluate(method_name, dataset_name):
    data_path = os.path.join(script_path, '../output/output_'+method_name+'/'+dataset_name+'/')
    true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq_'+dataset_name+'/')
    # figure_output_path = os.path.join(script_path, '../output/output_'+method_name+'/'+dataset_name+'/figs')
    figure_output_path = os.path.join(script_path, '../output/output_'+method_name+'/figs')
    matrix_save_path = os.path.join(script_path, '../output/output_'+method_name+'/'+dataset_name+'/matrix_RQ3')

    if not os.path.exists(figure_output_path):
        os.makedirs(figure_output_path)

    # draw segments
    fig, ax = plt.subplots(nrows=num*2)
    for i in range(num):
        prediction = reorder_label(np.load(os.path.join(data_path, 'state_seq/test'+str(i)+'.npy')))
        groundtruth = reorder_label(np.load(true_state_seq_path+'test'+str(i)+'.npy')[:len(prediction)])
        ax[i].imshow(groundtruth.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest')
        ax[i+num].imshow(prediction.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest')
        ax[i].set_xlabel('')
        ax[i+num].set_xlabel('')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i+num].set_xticks([])
        ax[i+num].set_yticks([])
    plt.savefig(os.path.join(figure_output_path,'seg_'+dataset_name+'.png'))
    plt.close()

    # draw matrixs
    groundtruth_matrix = np.load(os.path.join(matrix_save_path, 'groundtruth_matrix.npy'))
    prediction_matrix = np.load(os.path.join(matrix_save_path, 'prediction_matrix.npy'))
    # print(groundtruth_matrix.shape, prediction_matrix.shape)
    # fig, ax = plt.subplots(ncols=2, figsize=(8,4))
    # ax[0].imshow(groundtruth_matrix)
    # ax[1].imshow(prediction_matrix)
    # plt.savefig(os.path.join(figure_output_path,'mat_'+dataset_name+'.png'))
    # plt.close()

    precision, recall, threshold = precision_recall_curve(groundtruth_matrix.flatten(), prediction_matrix.flatten())
    f1 = 2*precision*recall/(precision+recall)
    f1[np.isnan(f1)]=0
    idx = np.argmax(f1)
    print('Best (F1-score, P, R) of '+method_name+' on '+dataset_name+': ', f1[idx], precision[idx], recall[idx])

    # draw PRC curve
    plt.style.use('classic')
    plt.grid()
    plt.plot(precision, recall, lw=2)
    plt.xlabel('Recall',fontsize=18)
    plt.ylabel('Precision',fontsize=18)
    plt.xticks([0.2,0.4,0.6,0.8,1.0],fontsize=14)
    plt.yticks([0.2,0.4,0.6,0.8,1.0],fontsize=14)
    plt.savefig(os.path.join(figure_output_path,'prc_'+dataset_name+'_RQ3.png'))
    plt.close()

    return groundtruth_matrix, prediction_matrix

def evaluate_RQ3():
    precision_list = []
    recall_list = []
    for method_name in method_list:
        fig_output_path= os.path.join(script_path, '../output/output_'+method_name+'/figs')
        if not os.path.exists(fig_output_path):
            os.makedirs(fig_output_path)
        gt_list = []
        p_list = []
        for i in range(1, num+1):
            print(method_name, i)
            groundtruth_matrix, prediction_matrix = evaluate(method_name, 'dataset'+str(i))
            gt_list.append(groundtruth_matrix.flatten())
            p_list.append(prediction_matrix.flatten())
        groundtruth = np.concatenate(gt_list)
        prediction = np.concatenate(p_list)
        print(groundtruth.shape, prediction.shape)
        precision, recall, threshold = precision_recall_curve(groundtruth.flatten(), prediction.flatten())
        f1 = 2*precision*recall/(precision+recall)
        precision_list.append(precision)
        recall_list.append(recall)
        f1[np.isnan(f1)]=0
        idx = np.argmax(f1)
        print('Best (F1-score, P, R) of '+method_name+': ', f1[idx], precision[idx], recall[idx])

    # draw PRC curve
    plt.style.use('classic')
    plt.grid()
    for i, method_name in enumerate(method_list):
        print(i)
        plt.plot(precision_list[i], recall_list[i], lw=2, label=method_name)
    plt.xlabel('Recall',fontsize=18)
    plt.ylabel('Precision',fontsize=18)
    plt.xticks([0.2,0.4,0.6,0.8,1.0],fontsize=14)
    plt.yticks([0.2,0.4,0.6,0.8,1.0],fontsize=14)
    plt.legend()
    plt.savefig(os.path.join(fig_output_path,'prc_RQ3.png'))
    plt.close()

# evaluate('StateCorr', 'dataset4')
evaluate_RQ3()