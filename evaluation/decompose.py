import numpy as np
import os

use_data = 'dataset5'

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../output/'+use_data+'/')
true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq_'+use_data+'/')
data_path = os.path.join(script_path, '../output/'+use_data+'/')

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
    # return np.log(1)
    # scores = p_xy*np.log(p_y_given_x/p_x)
    scores = p_xy*p_y_given_x/p_x
    print(p_x, p_xy, p_y_given_x, scores)
    return scores

score(np.array([0,0,0,1,1,0,0]), np.array([0,0,0,1,1,0,0]))

decomposed_state_seq_list = []
for i in range(2):
    state_seq = np.load(os.path.join(data_path, 'state_seq/test'+str(i)+'.npy'))
    decomposed_state_seq_list.append(decompose_state_seq(state_seq))

for sseq1 in decomposed_state_seq_list[0]:
    for sseq2 in decomposed_state_seq_list[1]:
        score(sseq1, sseq2)
