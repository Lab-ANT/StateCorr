import numpy as np
import os
import matplotlib.pyplot as plt
from TSpy.corr import partial_state_corr, lagged_partial_state_corr
from TSpy.label import reorder_label
from sklearn.metrics import precision_recall_curve

use_data = 'dataset4'

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../output/'+use_data+'/')
true_state_seq_path = os.path.join(script_path, '../data/synthetic_data/state_seq_'+use_data+'/')
figure_output_path = os.path.join(script_path, '../output/figs')

num = 20

def find_match(X, Y, score_matrix):
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
    matched_pair = find_match(groundtruth, prediction, score_matrix)
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

prediction_list = []
matched_list = []
gt_list = []

fig, ax = plt.subplots(nrows=num*2)
for i in range(num):
    groundtruth = reorder_label(np.load(true_state_seq_path+'test'+str(i)+'.npy'))
    gt_list.append([(s,s) for s in range(len(set(groundtruth)))])
    prediction = reorder_label(np.load(os.path.join(data_path, 'state_seq/test'+str(i)+'.npy')))
    # [i*2], [i*2+1]
    ax[i].imshow(groundtruth.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest')
    ax[i+num].imshow(prediction.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest')
    ax[i].set_xlabel('')
    ax[i+num].set_xlabel('')
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i+num].set_xticks([])
    ax[i+num].set_yticks([])
    matched_pairs = match_index(groundtruth, prediction)
    matched_list.append(matched_pairs)
    prediction_list.append(prediction)
plt.savefig(os.path.join(figure_output_path,'seg.png'))
plt.close()

def calculate_f1(G,P):
    U = list(set(G+P))
    TP, FP, FN = 0,0,0
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
    return f1, recall, precision

f1_list=[]
r_list=[]
p_list=[]
p_matrix_list = []
o_matrix_list = []
lag_matrix_list = []
true_lw = []
for i in range(num):
    for j in range(num):
        if i<=j:
            continue
        matrix, lag_matrix = lagged_partial_state_corr(prediction_list[i], prediction_list[j])
        true_lw.append((len(gt_list[i]), len(gt_list[j])))
        o_matrix_list.append(matrix)
        adjusted_matrix = match_matrix(matrix, matched_list[i], matched_list[j], len(gt_list[i]))
        p_matrix_list.append(adjusted_matrix)
        lag_matrix_list.append(lag_matrix)
        
        prediction = retrieve_relation(adjusted_matrix)
        f1, r, p = calculate_f1(gt_list[i], prediction)
        f1_list.append(f1)
        r_list.append(r)
        p_list.append(p)
        print(f1, r, p)
print('mean', np.mean(f1_list),np.mean(r_list),np.mean(p_list))

width_list = [max(m.shape[0],m.shape[1]) for m in p_matrix_list]
width = np.sum(width_list)
print(width)

groundtruth = np.zeros((width, width))
prediction = np.zeros((width,width))
omat = np.zeros((width,width))
lag = np.zeros((width,width))
start_row=0
start_col=0
for matrix,lagmat,lw in zip(p_matrix_list, lag_matrix_list, true_lw):
    prediction[start_row:start_row+matrix.shape[0],start_col:start_col+matrix.shape[1]]=matrix
    # omat[start_row:start_row+matrix.shape[0],start_col:start_col+matrix.shape[1]]=omatrix
    # lag[start_row:start_row+matrix.shape[0],start_col:start_col+matrix.shape[1]]=lagmat
    # print(lw)
    for i in range(lw[0]):
        groundtruth[start_row+i, start_col+i] = True
    start_row+=max(matrix.shape[0],matrix.shape[1])
    start_col+=max(matrix.shape[0],matrix.shape[1])

fig, ax = plt.subplots(ncols=3, figsize=(9,3))
ax[0].imshow(groundtruth)
ax[1].imshow(prediction)
ax[2].imshow(omat)
plt.savefig(os.path.join(figure_output_path,'mat.png'))
plt.close()

# print('f1',f1_score(groundtruth.flatten(), (prediction>=0.45).flatten()))
# print('p',precision_score(groundtruth.flatten(), (prediction>=0.45).flatten()))
# print('r',recall_score(groundtruth.flatten(), (prediction>=0.45).flatten()))
# fpr, tpr, thread = roc_curve(groundtruth.flatten(), prediction.flatten())
# print(auc(fpr, tpr))
# plt.plot(fpr, tpr, color = 'darkorange')
# plt.savefig(os.path.join(figure_output_path,'roc.png'))
# plt.close()

precision, recall, threshold = precision_recall_curve(groundtruth.flatten(), prediction.flatten())
f1 = 2*precision*recall/(precision+recall)
idx = np.argmax(f1)
print('max', f1[idx], precision[idx], recall[idx])
plt.style.use('classic')
plt.grid()
plt.plot(precision, recall, lw=2)
plt.xlabel('Recall',fontsize=18)
plt.ylabel('Precision',fontsize=18)
plt.xticks([0.2,0.4,0.6,0.8,1.0],fontsize=14)
plt.yticks([0.2,0.4,0.6,0.8,1.0],fontsize=14)
plt.savefig(os.path.join(figure_output_path,'prc.png'))
plt.close()