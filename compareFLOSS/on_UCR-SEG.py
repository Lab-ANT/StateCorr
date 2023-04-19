import sys
import os
sys.path.append('./')
from Time2State.time2state import Time2State
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *
from TSpy.utils import *
from scipy import io
import matplotlib.pyplot as plt

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/')
result_save_path = os.path.join(script_path, '../segment_results')

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def exp_on_UCR_SEG(win_size, step, verbose=False):
    output_path = os.path.join(result_save_path,'UCR-SEG/Time2State')
    create_path(output_path)
    params_LSE['in_channels'] = 1
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['out_channels'] = 2
    params_LSE['nb_steps'] = 20
    params_LSE['compared_length'] = win_size
    params_LSE['kernel_size'] = 3
    dataset_path = os.path.join(data_path,'UCR-SEG/')
    for fname in os.listdir(dataset_path):
        info_list = fname[:-4].split('_')
        f = info_list[0]
        print(f)
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
        prediction = t2s.state_seq
        np.save(os.path.join(output_path, fname), prediction)
        # ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        # prediction = np.array(prediction, dtype=int)
        # result = np.vstack([groundtruth, prediction])
        # np.save(os.path.join(output_path,fname[:-4]), result)
        # score_list.append(np.array([ari, anmi, nmi]))
        # plot_mulvariate_time_series_and_label_v2(data,groundtruth,prediction)
        # plt.savefig('1.png')
    #     if verbose:
    #         print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' %(fname, ari, anmi, nmi))
    # score_list = np.vstack(score_list)
    # print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' %(np.mean(score_list[:,0])\
    #     ,np.mean(score_list[:,1])
    #     ,np.mean(score_list[:,2])))

exp_on_UCR_SEG(256, 50)