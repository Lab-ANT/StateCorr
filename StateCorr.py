import sys
import tqdm
sys.path.append('./')
from Time2State.time2state import Time2State
from Time2State.adapers import *
from Time2State.clustering import *
from Time2State.default_params import *
from TSpy.utils import normalize
from TSpy.corr import state_correlation, lagged_state_correlation

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
        params_LSE['win_type'] = 'hanning'

        state_seq_list = []
        for i in tqdm.tqdm(range(num_ts)):
            t2s = Time2State(win_size, step, CausalConv_LSE_Adaper(params_LSE), DPGMM(None)).fit(normalize(X[i,:].reshape(-1,1)), win_size, step)
            state_seq_list.append(t2s.state_seq)
        self.__state_seq_list = np.array(state_seq_list)
        return self

    def calculate_matrix(self):
        self.__corr_matrix = state_correlation(self.__state_seq_list)
        # self.__corr_matrix, self.__lag_matrix = lagged_state_correlation(self.__state_seq_list)
        return self

    def load_state_seq_list(self, X):
        self.__state_seq_list = X

    @property
    def state_seq_list(self):
        return self.__state_seq_list
    
    @property
    def corr_matrix(self):
        return self.__corr_matrix

    @property
    def lag_matrix(self):
        return self.__lag_matrix

    def fit(self):
        pass

    def predict(self):
        pass