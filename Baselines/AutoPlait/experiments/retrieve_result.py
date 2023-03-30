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
    for j in range(20):
        prediction = read_result_dir2(
            os.path.join(result_path, 'synthetic_data/dataset%d/dat%d/'%(i,j+1)), 20000)#[:-1]
        # print(i,j,prediction.shape)
        # print(output_path)
        # print(os.path.join(output_path, '/dataset%d/state_seq/test%d.npy'%(i,j)))
        # print(output_path+'/dataset'+str(i)+'/state_seq/test'+str(j)+'.npy')
        np.save(output_path+'/dataset'+str(i)+'/state_seq/test'+str(j)+'.npy', prediction)


# data_path = os.path.join(script_path, '../../../data/')
# output_path = os.path.join(script_path, '../output/')
# redirect_path = os.path.join(script_path, '../../../results/output_AutoPlait')

# def create_path(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def read_result_dir2(path,length):
#     results = os.listdir(path)
#     results = [s if re.match('segment.\d', s) != None else None for s in results]
#     results = np.array(results)
#     idx = np.argwhere(results!=None)
#     results = results[idx].flatten()

#     label = np.zeros(length,dtype=int)

#     l = 0
#     for r in results:
#         data = pd.read_csv(path+r, names=['col0','col1'], header=None, index_col=False, sep = ' ')
#         start = data.col0
#         end = data.col1
#         for s, e in zip(start,end):
#             label[s:e+1]=l
#         l+=1
#     return label
   
# def read_result_dir(path,original_file_path):
#     results = os.listdir(path)
#     results = [s if re.match('segment.\d', s) != None else None for s in results]
#     results = np.array(results)
#     idx = np.argwhere(results!=None)
#     results = results[idx].flatten()

#     length = len_of_file(original_file_path)
#     # print(length)
#     label = np.zeros(length,dtype=int)

#     l = 0
#     for r in results:
#         data = pd.read_csv(path+r, names=['col0','col1'], header=None, index_col=False, sep = ' ')
#         start = data.col0
#         end = data.col1
#         for s, e in zip(start,end):
#             label[s:e+1]=l
#         l+=1
#     return label

# def redirect_USC_HAD():
#     data_redirect_path = os.path.join(redirect_path, 'USC-HAD')
#     create_path(data_redirect_path)
#     for subject in range(1,15):
#         for target in range(1,6):
#             _, groundtruth = load_USC_HAD(subject, target, data_path)
#             data_idx = (subject-1)*5+target
#             prediction = read_result_dir2(os.path.join(output_path, '_out_USC-HAD/dat'+str(data_idx)+'/'),len(groundtruth))
#             ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
#             print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' %(data_idx, ari, anmi, nmi))
#             print(os.path.join(data_redirect_path,'s%d_t%d'%(subject,target)))
#             prediction = np.array(prediction, dtype=int)
#             result = np.vstack([groundtruth, prediction])
#             np.save(os.path.join(data_redirect_path,'s%d_t%d'%(subject,target)), result)

# def redirect_MoCap():
#     data_redirect_path = os.path.join(redirect_path, 'MoCap')
#     create_path(data_redirect_path)
#     for n,i in zip(['01','02','03','07','08','09','10','11','14'], range(1,10)):
#         prediction = read_result_dir(os.path.join(output_path, '_out_MoCap/dat'+str(i)+'/'), os.path.join(data_path, 'MoCap/4d/amc_86_'+n+'.4d'))
#         groundtruth = seg_to_label(dataset_info['amc_86_'+str(n)+'.4d']['label'])
#         fname = 'amc_86_'+str(n)+'.4d'
#         prediction = np.array(prediction, dtype=int)
#         result = np.vstack([groundtruth, prediction])
#         np.save(os.path.join(data_redirect_path, fname), result)

# def redirect_ActRecTut():
#     data_redirect_path = os.path.join(redirect_path, 'ActRecTut')
#     create_path(data_redirect_path)
#     for i, s in enumerate(['subject1_walk', 'subject2_walk']):
#         full_path = data_path+'/ActRecTut/data_for_AutoPlait/'+s+'_groundtruth.txt'
#         groundtruth = np.loadtxt(full_path)
#         prediction = read_result_dir2(output_path+'/_out_ActRecTut/dat'+str(i+1)+'/', len(groundtruth))
#         prediction = np.array(prediction, dtype=int)
#         result = np.vstack([groundtruth, prediction])
#         np.save(os.path.join(data_redirect_path, s), result)

# def redirect_UCR_SEG():
#     data_redirect_path = os.path.join(redirect_path, 'UCR-SEG')
#     create_path(data_redirect_path)
#     base = os.path.join(data_path, 'UCR-SEG/UCR_datasets_seg/')
#     f_list = os.listdir(base)
#     for fname,n in zip(f_list, range(1,len(f_list)+1)):
#         prediction = read_result_dir(os.path.join(output_path, '_out_UCR_SEG/dat'+str(n)+'/'), base+fname)
#         info_list = fname[:-4].split('_')
#         f = info_list[0]
#         seg_info = {}
#         i = 0
#         for seg in info_list[2:]:
#             seg_info[int(seg)]=i
#             i+=1
#         seg_info[len_of_file(base+fname)]=i
#         groundtruth = seg_to_label(seg_info)[:]
#         prediction = np.array(prediction, dtype=int)
#         result = np.vstack([groundtruth, prediction])
#         np.save(os.path.join(data_redirect_path, fname), result)

# def redirect_PAMAP2():
#     data_redirect_path = os.path.join(redirect_path, 'PAMAP2')
#     create_path(data_redirect_path)
#     for i in range(1,10):
#         groundtruth = np.loadtxt(data_path+'/PAMAP2/data_for_AutoPlait/groundtruth_subject10'+str(i)+'.txt')#[::10]
#         prediction = read_result_dir(output_path+'/_out_PAMAP2/dat'+str(i)+'/', data_path+'/PAMAP2/data_for_AutoPlait/subject10'+str(i)+'.txt')#[::10]
#         prediction = np.array(prediction, dtype=int)
#         result = np.vstack([groundtruth, prediction])
#         np.save(os.path.join(data_redirect_path, '10'+str(i)), result)

# def redirect_synthetic():
#     data_redirect_path = os.path.join(redirect_path, 'synthetic')
#     create_path(data_redirect_path)
#     base = os.path.join(data_path, 'synthetic_data_for_Autoplait/')
#     f_list = os.listdir(base)
#     f_list.remove('list')
#     f_list = np.sort(f_list)
#     for fname in f_list:
#         n = int(fname[4:-4])
#         data = pd.read_csv(base+'test'+str(n)+'.csv', sep=' ', usecols=range(4)).to_numpy()
#         prediction = read_result_dir(os.path.join(output_path, '_out_synthetic/dat'+str(n+1)+'/'),base+fname)[:-1]
#         groundtruth = pd.read_csv(base+'test'+str(n)+'.csv', sep=' ', usecols=[4]).to_numpy().flatten()
#         prediction = np.array(prediction, dtype=int)
#         result = np.vstack([groundtruth, prediction])
#         np.save(os.path.join(data_redirect_path, fname[4:-4]), result)

# redirect_ActRecTut()
# redirect_synthetic()
# redirect_UCR_SEG()
# redirect_PAMAP2()
# redirect_MoCap()
# redirect_USC_HAD()