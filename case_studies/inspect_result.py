import os
import numpy as np
import matplotlib.pyplot as plt
from TSpy.label import adjust_label, compact
from TSpy.dataset import load_SMD

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/SMD/')

use_data = 'machine-1-6'

# matrix_path = os.path.join(script_path, '../case_study1/pearson/'+use_data+'.npy')
matrix_path = os.path.join(script_path, '../case_study1/state/'+use_data+'.npy')
state_seq_path = os.path.join(script_path, '../case_study1/state_seq/'+use_data+'.npy')

def decompose_state_seq(X):
    state_set = set(list(X))
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

def find_best_match(X, Y, score_matrix):
    # print(score_matrix)
    matched_pair = []
    height, width = score_matrix.shape
    for i in range(min(height, width)):
        row, col = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
        matched_pair.append((row, col))
        # print(score_matrix, row, col)
        score_matrix[row,:] = 0
        score_matrix[:,col] = 0
        if np.sum(score_matrix)==0:
            break
    # print(set(X), set(Y))
    print(compact(X), compact(Y))
    print(matched_pair)
    new_X = X.copy()
    new_Y = Y.copy()+4
    # color = 0
    for i,j in matched_pair:
        # new_X[np.argwhere(X==i)]=color
        new_Y[np.argwhere(Y==j)]=i
        # color+=1
    # new_Y[new_Y>=10]=-1
    # print(set(new_X), set(new_Y))
    print(compact(new_X), compact(new_Y))
    print('========================')
    return X, new_Y

def exclude_outlier(X):
    mean = np.mean(X)
    idx = np.argwhere(X>=3*mean)
    X[idx] = mean
    idx = np.argwhere(X<=mean/3)
    X[idx] = mean
    return X

def find_top_k(id, k):
    data,_,_,_,_,_ = load_SMD(data_path, use_data)
    data = data[::4]
    corr_matrix = np.load(matrix_path)
    state_seq_array = np.load(state_seq_path)
    col = corr_matrix[:,id]
    idx = np.argsort(-col)
    print(idx)

    # fig, ax = plt.subplots(nrows=k, sharex=True, figsize=(4,4.5))
    fig, ax = plt.subplots(nrows=k, sharex=True, figsize=(8,4.5))
    state_seq1 = adjust_label(state_seq_array[idx[0]]).flatten()
    for i in range(k):
        data_ = data[:,idx[i]]
        state_seq = adjust_label(state_seq_array[idx[i]]).flatten()#.reshape(1,-1)
        matrix = partial_state_corr(state_seq1, state_seq)
        matrix = np.round(matrix, 2)
        state_seq_new, state_seq = find_best_match(state_seq1, state_seq, matrix)
        state_seq = state_seq.reshape(1,-1)
        state_seq = np.concatenate([state_seq, state_seq])
        # ax[i].plot(exclude_outlier(data_), color= '#348ABC')
        ax[i].imshow(state_seq, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.6, origin='lower', vmax=5, vmin=0)
        ax[i].plot(data_, lw=0.5)
        ax[i].set_ylim([-0.1,1.1])

        if i == 0:
            ax[i].set_title('Indicator %d'%(i+1), loc='left', y=0.55, x=0.02,
                fontsize='medium')
        elif i == 1:
            ax[i].set_title('%d-st state-correlated indicator'%(i), loc='left', y=0.55, x=0.02,
                fontsize='medium')
        elif i == 2:
            ax[i].set_title('%d-nd state-correlated indicator'%(i), loc='left', y=0.55, x=0.02,
                fontsize='medium')
        elif i == 3:
            ax[i].set_title('%d-rd state-correlated indicator'%(i), loc='left', y=0.55, x=0.02,
                fontsize='medium')
        else:
            ax[i].set_title('%d-th state-correlated indicator'%(i), loc='left', y=0.55, x=0.02,
                fontsize='medium')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)
    plt.savefig('look.png')
    print(idx[:k])

find_top_k(21, 6)
# find_top_k(18, 5)