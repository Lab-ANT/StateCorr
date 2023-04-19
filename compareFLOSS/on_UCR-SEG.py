import sys
import os
sys.path.append('./')
from Time2State.time2state import Time2State
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *
from TSpy.utils import *
from TSpy.label import seg_to_label, reorder_label, compact
from scipy import io
import matplotlib.pyplot as plt

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/')

def exp_on_UCR_SEG(win_size, step, verbose=False):
    score_list = []
    out_path = os.path.join(output_path,'UCR-SEG3')
    create_path(out_path)
    params_LSE['in_channels'] = 1
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['out_channels'] = 2
    params_LSE['nb_steps'] = 20
    params_LSE['compared_length'] = win_size
    params_LSE['kernel_size'] = 3
    dataset_path = os.path.join(data_path,'UCR-SEG/UCR_datasets_seg/')
    for fname in os.listdir(dataset_path):
        info_list = fname[:-4].split('_')
        # f = info_list[0]
        window_size = int(info_list[1])
        seg_info = {}
        i = 0
        for seg in info_list[2:]:
            seg_info[int(seg)]=i
            i+=1
        seg_info[len_of_file(dataset_path+fname)]=i
        num_state=len(seg_info)
        df = pd.read_csv(dataset_path+fname)
        data = df.to_numpy()
        data = normalize(data)
        t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
        # t2s = Time2State(win_size, step, TS2Vec_Adaper(params_TS2Vec), DPGMM(None, alpha=None)).fit(data, win_size, step)
        groundtruth = seg_to_label(seg_info)[:-1]
        prediction = t2s.state_seq
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path,fname[:-4]), result)
        score_list.append(np.array([ari, anmi, nmi]))
        # plot_mulvariate_time_series_and_label_v2(data,groundtruth,prediction)
        # plt.savefig('1.png')
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
        ,np.mean(score_list[:,1])
        ,np.mean(score_list[:,2])))

def load_ActRecTut(use_data):
    dir_name = use_data
    dataset_path = os.path.join(data_path,'ActRecTut/'+dir_name+'/data.mat')
    data = io.loadmat(dataset_path)
    groundtruth = data['labels'].flatten()
    groundtruth = reorder_label(groundtruth)
    data = data['data'][:,0:10]
    return data, groundtruth

win_size = 256
step = 50

params_LSE['M'] = 10
params_LSE['N'] = 4
params_LSE['out_channels'] = 2
params_LSE['nb_steps'] = 20
params_LSE['compared_length'] = win_size
params_LSE['kernel_size'] = 3

data, groundtruth = load_ActRecTut('subject1_walk')
params_LSE['in_channels'] = data.shape[1]
t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)

plt.style.use('classic')
fig, ax = plt.subplots(nrows=4)
ax[0].plot(data)
ax[0].set_xlim([0, len(data)])
ax[1].plot(t2s.velocity)
ax[1].set_xlim([0, len(t2s.velocity)])
ax[1].hlines(np.mean(t2s.velocity), 0, len(t2s.velocity), color="red")
ax[2].imshow(groundtruth.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')
ax[3].imshow(t2s.state_seq.reshape(-1,1).T, aspect='auto', cmap='tab20c', interpolation='nearest', alpha=0.5, origin='lower')

max_v = np.max(t2s.velocity)
min_v = np.min(t2s.velocity)
for cut in t2s.change_points:
    ax[1].vlines(cut, min_v, max_v,color="red")

plt.savefig('velocity.png')