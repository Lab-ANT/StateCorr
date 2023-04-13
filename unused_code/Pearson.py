from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import numpy as np

data = np.random.randint(100,size=400).reshape(200,-1)
print(data.shape)

result = grangercausalitytests(data, maxlag=2, verbose=False)
print(result)