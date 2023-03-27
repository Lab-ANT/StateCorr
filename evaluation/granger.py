from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import numpy as np

# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
# print(df)
# df['month'] = df.date.dt.month
# print(df)
data = np.random.randint(100,size=400).reshape(200,-1)
print(data.shape)

grangercausalitytests(data, maxlag=2)