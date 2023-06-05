from TSpy.utils import normalize
import sys
sys.path.append('./')
from Time2State.time2state import Time2State
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *
import tqdm

class StateCorr():
    def __init__(self) -> None:
        pass

    def fit_predict(self, X, win_size, step):
        '''
        X is of shape (B, C, L)
        '''

        win_size = win_size
        step = step

        num_ts = X.shape[0]
        num_channels = X.shape[2]

        params_LSE['in_channels'] = num_channels
        params_LSE['out_channels'] = 1
        params_LSE['nb_steps'] = 40
        params_LSE['win_size'] = win_size
        # params_LSE['win_type'] = 'hanning'

        state_seq_list = []
        for i in tqdm.tqdm(range(num_ts)):
            t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(normalize(X[i,:].reshape(-1,1)), win_size, step)
            state_seq_list.append(t2s.state_seq)
        state_seq_list = np.array(state_seq_list)
        return state_seq_list

    def fit(self):
        pass

    def predict(self):
        pass