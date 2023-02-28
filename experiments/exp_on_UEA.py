import os
from TSpy.dataset import *
import sys
import os
from TSpy.eval import *
from TSpy.label import *
from TSpy.utils import *
from TSpy.view import *
from TSpy.dataset import *

sys.path.append('./')
from Time2State.time2state import Time2State
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *

UEA_SEG_path = os.path.join(os.path.dirname(__file__), '../../data/UEA_SEG/')

dataset_list = os.listdir(UEA_SEG_path)
dataset_list.sort()

for dataset in dataset_list:
    print(dataset)
    data, label = load_general_seg_dagaset(os.path.join(UEA_SEG_path, dataset))
    win_size = 512
    step = 50
    params_LSE['in_channels'] = data.shape[1]
    params_LSE['M'] = 10
    params_LSE['N'] = 4
    params_LSE['out_channels'] = 2
    params_LSE['nb_steps'] = 20
    params_LSE['compared_length'] = win_size
    params_LSE['kernel_size'] = 3
    data = normalize(data)
    t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(data, win_size, step)
    prediction = t2s.state_seq
    ari, anmi, nmi = evaluate_clustering(label, prediction)
    print(ari, anmi, nmi)