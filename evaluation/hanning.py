import math
import numpy as np
import torch

x = np.arange(0,20*math.pi,0.001)
v = np.cos(x)
length = len(x)
print(v)

# print(length)
data = torch.ones(16,4,length)

def hanning_tensor(X):
    length = X.size(2)
    print(length)
    weight = (1-np.cos(2*math.pi*np.arange(length)/length))/2
    weight

hanning_tensor(data)
# def hanning(X):
#     length = len(X)
#     return (1-np.cos(2*math.pi*np.arange(length)/length))/2

# import matplotlib.pyplot as plt
# plt.plot(hanning(x)*v)
# plt.show()