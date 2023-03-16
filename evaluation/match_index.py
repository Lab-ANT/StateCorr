import numpy as np
import os
from TSpy.corr import decompose_state_seq, add_lag, score, partial_state_corr

use_data = 'dataset5'

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../output/'+use_data+'/')
true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq_'+use_data+'/')

if not os.path.exists(data_path+'matrix'):
    os.makedirs(data_path+'matrix')

def lagged_partial_state_corr(X, Y, atom_step=0.001, max_ratio=0.05):
    length = len(X)
    k = int(max_ratio/atom_step)
    listX = decompose_state_seq(X)
    listY = decompose_state_seq(Y)
    score_matrix = np.zeros((len(listX),len(listY)))
    # lag_matrix = np.zeros((len(listX),len(listY)))
    for i in range(len(listX)):
        for j in range(len(listY)):
            sssX = listX[i]
            sssY = listY[j]
            max_score = -1
            lag = 0
            for l in range(-k,k+1):
                lag_len = int(l*atom_step*length)
                sX, sY = add_lag(sssX, sssY, lag_len)
                Jscore = score(sX, sY)
                if Jscore>=max_score:
                    max_score=Jscore
                    lag=lag_len
            score_matrix[i,j] = max_score
    # print(np.round(score_matrix, 2))
    return score_matrix, None

def find_match(X, Y, score_matrix):
    matched_pair = {}
    height, width = score_matrix.shape
    for i in range(min(height, width)):
        row, col = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        # matched_pair.append((row, col))
        matched_pair[col]=row
        score_matrix[row,:] = 0
        score_matrix[:,col] = 0
        if np.sum(score_matrix)==0:
            break
    return matched_pair

def match_index(groundtruth,prediction):
    score_matrix = partial_state_corr(groundtruth, prediction)
    # print(np.round(score_matrix, 2))
    matched_pair = find_match(groundtruth, prediction, score_matrix)
    # print(matched_pair)
    return matched_pair

def retrieve_relation(matrix, x, y):
    relation_set = []
    for i in range(matrix.shape[0]):
        idx = np.argmax(matrix[i,:])
        if i in x and idx in y:
            relation_set.append((x[i],y[idx]))
        else:
            relation_set.append((i,idx))
    # print(relation_set)
    return relation_set

prediction_list = []
matched_list = []

from TSpy.label import reorder_label
for i in range(4):
    groundtruth = reorder_label(np.load(true_state_seq_path+'test'+str(i)+'.npy'))
    prediction = reorder_label(np.load(os.path.join(data_path, 'state_seq/test'+str(i)+'.npy')))
    matched_pairs = match_index(groundtruth, prediction)
    matched_list.append(matched_pairs)
    prediction_list.append(prediction)

gt = [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)]

def calculate_f1(G,P):
    U = list(set(G+P))
    TP, FP, FN = 0,0,0
    # print(U)
    for r in U:
        if r in G and r in P:
            TP+=1
        elif r in G and r not in P:
            FN+=1
        elif r in P and r not in G:
            FP+=1
    recall=TP/(TP+FN)
    precision=TP/(TP+FP)
    f1=2*recall*precision/(recall+precision)
    print(f1, recall, precision)



for i in range(4):
    for j in range(4):
        if i<j:
            continue
        matrix, lag_matrix = lagged_partial_state_corr(prediction_list[i], prediction_list[j])
        # print(matched_list[i], matched_list[j])
        prediction = retrieve_relation(matrix, matched_list[i], matched_list[j])
        calculate_f1(gt, prediction)


# for i in range(2):
#     # groundtruth = np.load(true_state_seq_path+'test'+str(i)+'.npy')
#     prediction = np.load(os.path.join(data_path, 'state_seq/test'+str(i)+'.npy'))
#     prediction_list.append(prediction)
#     # lagged_partial_state_corr(groundtruth, prediction)
#     # print('original',normalized_mutual_info_score(groundtruth, prediction))
#     # groundtruth, prediction = match_index(groundtruth, prediction)
#     # print('matched',normalized_mutual_info_score(groundtruth, prediction))