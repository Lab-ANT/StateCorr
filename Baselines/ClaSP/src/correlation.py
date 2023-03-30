import os, sys
sys.path.append("Baselines/ClaSP/")

import numpy as np
# np.random.seed(1379)

import tqdm
from TSpy.label import *
from TSpy.utils import *
from TSpy.dataset import *

from src.clasp import extract_clasp_cps_from_multivariate_ts

import os
from tslearn.clustering import TimeSeriesKMeans

''' Define relative path. '''
script_path = os.path.dirname(__file__)

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def padding(X, max_length):
    if len(X.shape) == 1:
        dim = 1
        X = np.expand_dims(X, axis=1)
    else:
        _, dim = X.shape
    data = np.zeros((max_length, dim))
    length = len(X)
    data[:length] = X
    return data

def padding_and_stack(seg_list):
    max_length = 0
    for seg in seg_list:
        length = len(seg)
        if length > max_length:
            max_length = length
    new_seg_list = []
    for seg in seg_list:
        new_seg_list.append(padding(seg, max_length))
    result = np.stack(new_seg_list)
    return result

def calculate_seg_len_list(found_cps,length):
    length_list = []
    prev = 0
    for cp in found_cps:
        length_list.append(int(cp-prev))
        prev=cp
    length_list.append(length-prev)
    return length_list

def cluster_segs(X, found_cps, n_states):
    seg_list = []
    length = len(X)
    start = 0
    for cp in found_cps:
        seg_list.append(X[start:cp])
        start = cp
    seg_list.append(X[start:length])
    segments = padding_and_stack(seg_list)
    # dtw, euclidean, softdtw
    ts_kmeans = TimeSeriesKMeans(n_clusters=n_states,metric='euclidean').fit(segments)
    labels = ts_kmeans.labels_
    seg_label_list = []
    length_list = calculate_seg_len_list(found_cps,length)

    for label, length in zip(labels, length_list):
        seg_label_list.append(label*np.ones(length))
    result = np.hstack(seg_label_list)
    return result

def run_clasp(X, window_size, num_cps, n_states, offset):
    profile_, found_cps, _ = extract_clasp_cps_from_multivariate_ts(X, window_size, num_cps, offset)
    found_cps.sort()
    prediction = cluster_segs(X, found_cps, n_states)
    return prediction

def use_ClaSP(use_data):
    data_path = os.path.join(script_path, '../../../data/synthetic_data/'+use_data)
    true_state_seq_path = os.path.join(script_path, '../../../data/synthetic_data/state_seq_'+use_data+'/')
    file_list = os.listdir(data_path)
    output_path = os.path.join(script_path, '../../../output/output_ClaSP/'+use_data+'/state_seq/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in tqdm.tqdm(file_list):
        data = np.load(os.path.join(data_path, file_name))
        state_seq = np.load(os.path.join(true_state_seq_path, file_name))
        n_states = len(set(state_seq))
        prediction = run_clasp(data, 50, 40, n_states, 0.02)
        prediction = prediction.astype(int)
        np.save(output_path+file_name[:-4]+'.npy', prediction)

for i in range(1,6):
    use_ClaSP('dataset'+str(i))