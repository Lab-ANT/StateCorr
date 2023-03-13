import numpy as np
import os

use_data = 'dataset5'

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../output/'+use_data+'/')
true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq_'+use_data+'/')
data_path = os.path.join(script_path, '../output/'+use_data+'/')

def add_lag(X, Y, lag):
    if lag>0:
        seq1 = X[lag:]
        seq2 = Y[:-lag]
    elif lag<0:
        seq1 = X[:lag]
        seq2 = Y[-lag:]
    else:
        seq1 = X
        seq2 = Y
    return seq1, seq2

def decompose_state_seq(X):
    state_set = set(X)
    # return state_set
    print(state_set)
    single_state_seq_list = []
    for state in list(state_set):
        single_state_seq = np.zeros(X.shape, dtype=int)
        single_state_seq[np.argwhere(X==state)]=1
        single_state_seq_list.append(single_state_seq)
    return np.array(single_state_seq_list)

def score(X,Y):
    length = len(X)
    p_x = np.count_nonzero(X)/length
    p_xy = np.sum((X+Y)==2)/length
    new = Y[np.argwhere(X==1)]
    p_y_given_x = np.count_nonzero(new)/len(new)
    # scores = p_xy*np.log(p_y_given_x/p_x)
    scores = p_xy*p_y_given_x/p_x
    # print(p_x, p_xy, p_y_given_x, scores)
    return scores

def partial_state_corr(X,Y):
    listX = decompose_state_seq(X)
    listY = decompose_state_seq(Y)
    score_matrix = np.zeros((len(listX),len(listY)))
    for i in range(len(listX)):
        for j in range(len(listY)):
            sssX = listX[i]
            sssY = listY[j]
            Jscore = score(sssX, sssY)
            score_matrix[i,j] = Jscore
    return score_matrix

# def match_label(X, Y, score_matrix):
#     print(score_matrix)
#     height, width = score_matrix.shape
#     print(height, width)
#     idx = np.argmax(score_matrix)
#     if idx%width==0:
#         row = int(idx/width)-1
#         col = width-1
#     else:
#         row = int(idx/width)
#         col = idx%width
#     # print(idx, row, col)
#     return X, Y

def match_label(X, Y, score_matrix):
    print(score_matrix)
    height, width = score_matrix.shape
    for i in range(height):
        idx = np.argmax(score_matrix[i,:])
        print(idx)
        adjust_idx = np.argwhere(Y==idx)
        Y[adjust_idx] = i
    return X, Y

def lagged_partial_state_corr(X, Y):
    listX = decompose_state_seq(X)
    listY = decompose_state_seq(Y)
    score_matrix = np.zeros((len(listX),len(listY)))
    for i in range(len(listX)):
        for j in range(len(listY)):
            sssX = listX[i]
            sssY = listY[j]
            Jscore = score(sssX, sssY)
            score_matrix[i,j] = Jscore
    return score_matrix

score(np.array([0,0,0,1,1,0,0]), np.array([0,0,0,1,1,0,0]))

state_seq_list = []
for i in range(20):
    state_seq = np.load(os.path.join(data_path, 'state_seq/test'+str(i)+'.npy'))
    state_seq_list.append(state_seq)

matrix = partial_state_corr(state_seq_list[0], state_seq_list[2])
# print(np.round(matrix, 2))
matrix = np.round(matrix, 2)
X, Y = match_label(state_seq_list[0], state_seq_list[2], matrix)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=2)
ax[0].imshow(X.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest',)
ax[1].imshow(Y.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest',)
ax[0].set_ylim([-0.5,0.5])
ax[1].set_ylim([-0.5,0.5])
plt.savefig('partial.png')