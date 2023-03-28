import numpy as np
import os
import matplotlib.pyplot as plt
from TSpy.corr import partial_state_corr, lagged_partial_state_corr
from TSpy.label import reorder_label
from sklearn.metrics import precision_recall_curve

num_in_group = 5
num = 5
method_list = ['StateCorr', 'TICC']
script_path = os.path.dirname(__file__)

def find_match(score_matrix):
    matched_pair = {}
    height, width = score_matrix.shape
    for i in range(min(height, width)):
        row, col = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        matched_pair[col]=row
        score_matrix[row,:] = 0
        score_matrix[:,col] = 0
        if np.sum(score_matrix)==0:
            break
    return matched_pair

def match_index(groundtruth,prediction):
    score_matrix = partial_state_corr(groundtruth, prediction)
    matched_pair = find_match(score_matrix)
    return matched_pair

def retrieve_relation(matrix):
    relation_set = []
    for i in range(matrix.shape[0]):
        idx = np.argmax(matrix[i,:])
        relation_set.append((i,idx))
    return relation_set

def match_matrix(matrix, x, y, lw):
    new_height = max(matrix.shape[0],lw)
    new_weight = max(matrix.shape[1],lw)
    new_matrix = np.zeros((new_height, new_weight))
    new_matrix[:matrix.shape[0],:matrix.shape[1]] = matrix

    pre=[i for i in x]
    post=[x[i] for i in pre]
    # print(pre,post,x)
    new_matrix[post,:] = new_matrix[pre,:]
    pre=[i for i in y]
    post=[y[i] for i in pre]
    # print(pre,post,x)
    new_matrix[:,post] = new_matrix[:,pre]
    return new_matrix

def evaluate(method_name, dataset_name):
    data_path = os.path.join(script_path, '../output/output_'+method_name+'/'+dataset_name+'/')
    true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq_'+dataset_name+'/')
    # figure_output_path = os.path.join(script_path, '../output/output_'+method_name+'/figs')
    matrix_save_path = os.path.join(script_path, '../output/output_'+method_name+'/matrix_RQ3')

    if not os.path.exists(matrix_save_path):
        os.makedirs(matrix_save_path)

    matched_list = []
    prediction_list = []
    gt_list = []

    for i in range(num):
        # for some methods, prediction may be shorter than ground truth
        # but the difference is extremely small, we can simply cut the ground truth
        prediction = reorder_label(np.load(os.path.join(data_path, 'state_seq/test'+str(i)+'.npy')))
        groundtruth = reorder_label(np.load(true_state_seq_path+'test'+str(i)+'.npy')[:len(prediction)])
        gt_list.append([(s,s) for s in range(len(set(groundtruth)))])
        # find matched pairs
        matched_pairs = match_index(groundtruth, prediction)
        matched_list.append(matched_pairs)
        prediction_list.append(prediction)

    p_matrix_list = []
    # lag_matrix_list = []
    true_lw = []
    for i in range(num):
        for j in range(num):
            if i<=j:
                continue
            matrix, lag_matrix = lagged_partial_state_corr(prediction_list[i], prediction_list[j])
            true_lw.append((len(gt_list[i]), len(gt_list[j])))
            adjusted_matrix = match_matrix(matrix, matched_list[i], matched_list[j], len(gt_list[i]))
            p_matrix_list.append(adjusted_matrix)
            # lag_matrix_list.append(lag_matrix)

    # calculate matrix width for every pair
    width_list = [max(m.shape[0],m.shape[1]) for m in p_matrix_list]
    width = np.sum(width_list)

    groundtruth_matrix = np.zeros((width, width))
    prediction_matrix = np.zeros((width,width))
    # lag_matrix = np.zeros((width,width))
    start_row=0
    start_col=0
    for matrix,lw in zip(p_matrix_list, true_lw):
        prediction_matrix[start_row:start_row+matrix.shape[0],start_col:start_col+matrix.shape[1]]=matrix
        # lag[start_row:start_row+matrix.shape[0],start_col:start_col+matrix.shape[1]]=lagmat
        # print(lw)
        for i in range(lw[0]):
            groundtruth_matrix[start_row+i, start_col+i] = True
        start_row+=max(matrix.shape[0],matrix.shape[1])
        start_col+=max(matrix.shape[0],matrix.shape[1])

    np.save(os.path.join(matrix_save_path, 'groundtruth_matrix.npy'), groundtruth_matrix)
    np.save(os.path.join(matrix_save_path, 'prediction_matrix.npy'), prediction_matrix)
    # fig, ax = plt.subplots(ncols=2, figsize=(9,3))
    # ax[0].imshow(groundtruth_matrix)
    # ax[1].imshow(prediction_matrix)
    # plt.savefig(os.path.join(figure_output_path,'mat.png'))
    # plt.close()

    # precision, recall, threshold = precision_recall_curve(groundtruth_matrix.flatten(), prediction_matrix.flatten())
    # f1 = 2*precision*recall/(precision+recall)
    # idx = np.argmax(f1)
    # print('max', f1[idx], precision[idx], recall[idx])
    # plt.style.use('classic')
    # plt.grid()
    # plt.plot(precision, recall, lw=2)
    # plt.xlabel('Recall',fontsize=18)
    # plt.ylabel('Precision',fontsize=18)
    # plt.xticks([0.2,0.4,0.6,0.8,1.0],fontsize=14)
    # plt.yticks([0.2,0.4,0.6,0.8,1.0],fontsize=14)
    # plt.savefig(os.path.join(figure_output_path,'prc.png'))
    # plt.close()

evaluate('StateCorr', 'dataset1')