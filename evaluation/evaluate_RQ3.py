import numpy as np
import os
from TSpy.corr import partial_state_corr, lagged_partial_state_corr
from TSpy.label import reorder_label
from TSpy.eval import ARI

num_in_group = 5
num = 5
# method_list = ['HDP-HSMM', 'ClaSP', 'AutoPlait', 'StateCorr', 'TICC']
method_list = ['TICC', 'ClaSP', 'HDP-HSMM']

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
    matrix_save_path = os.path.join(script_path, '../output/output_'+method_name+'/'+dataset_name+'/matrix_RQ3')

    if not os.path.exists(matrix_save_path):
        os.makedirs(matrix_save_path)

    matched_list = []
    prediction_list = []
    gt_list = []

    for i in range(num_in_group):
        # for some methods, prediction may be shorter than ground truth
        # but the difference is extremely small, we can simply cut the ground truth
        prediction = reorder_label(np.load(os.path.join(data_path, 'state_seq/test'+str(i)+'.npy')))
        groundtruth = reorder_label(np.load(true_state_seq_path+'test'+str(i)+'.npy')[:len(prediction)])
        print(ARI(prediction, groundtruth))
        gt_list.append([(s,s) for s in range(len(set(groundtruth)))])
        # find matched pairs
        matched_pairs = match_index(groundtruth, prediction)
        matched_list.append(matched_pairs)
        prediction_list.append(prediction)

    p_matrix_list = []
    # lag_matrix_list = []
    true_lw = []
    for i in range(num_in_group):
        for j in range(num_in_group):
            if i<=j:
                continue
            matrix, lag_matrix = lagged_partial_state_corr(prediction_list[i], prediction_list[j])
            # print(matrix)
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

def evaluate_RQ3():
    for method_name in method_list:
        for i in range(1, num+1):
            print(method_name, i)
            evaluate(method_name, 'dataset'+str(i))

# evaluate('TICC', 'dataset1')
evaluate_RQ3()