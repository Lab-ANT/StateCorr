import math
import numpy as np
import torch


x = np.arange(0,20*math.pi,0.001)
v = np.cos(x)+1
v = np.stack([v,v,v,v])
v = np.stack([v,v,v,v,v,v,v,v])
data = torch.tensor(v)
print(data.size())

def hanning_tensor(X):
    length = X.size(2)
    # print(length)
    weight = (1-np.cos(2*math.pi*np.arange(length)/length))/2
    weight = torch.tensor(weight)
    return weight*data

new_data = hanning_tensor(data)

import matplotlib.pyplot as plt
plt.plot(new_data[1,1,:])
plt.show()


# hanning_tensor(data)

# x = np.arange(0,20*math.pi,0.001)
# v = np.cos(x)+1
# length = len(x)
# print(v)

# def hanning(X):
#     length = len(X)
#     return (1-np.cos(2*math.pi*np.arange(length)/length))/2

# import matplotlib.pyplot as plt
# plt.plot(hanning(x)*v)
# plt.show()