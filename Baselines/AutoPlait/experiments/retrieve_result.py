import numpy as np
import os
import re
import pandas as pd

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_result_dir2(path,length):
    results = os.listdir(path)
    results = [s if re.match('segment.\d', s) != None else None for s in results]
    results = np.array(results)
    idx = np.argwhere(results!=None)
    results = results[idx].flatten()

    label = np.zeros(length,dtype=int)

    l = 0
    for r in results:
        data = pd.read_csv(path+r, names=['col0','col1'], header=None, index_col=False, sep = ' ')
        start = data.col0
        end = data.col1
        for s, e in zip(start,end):
            label[s:e+1]=l
        l+=1
    return label

script_path = os.path.dirname(__file__)
result_path = os.path.join(script_path, '../output/')
output_path = os.path.join(script_path, '../../../output/output_AutoPlait')

for i in range(1,6):
    result_save_path = os.path.join(output_path, 'dataset%d/state_seq'%(i))
    create_path(result_save_path)
    for j in range(5):
        prediction = read_result_dir2(
            os.path.join(result_path, 'synthetic_data/dataset%d/dat%d/'%(i,j+1)), 20000)#[:-1]
        np.save(output_path+'/dataset'+str(i)+'/state_seq/test'+str(j)+'.npy', prediction)