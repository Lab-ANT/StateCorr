import scipy.cluster.hierarchy as sch
import numpy as np
from TSpy.dataset import load_SMD
import os
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hac

def hierarchal_clustering(X):
    pairwise_distances = sch.distance.pdist(X)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    # idx = np.argsort(idx_to_cluster_array)
    # print(idx)
    print(idx_to_cluster_array)
    return idx_to_cluster_array

def  hierarchal_clustering2(X):
# for method, axes in zip(['single', 'complete'], axes23):
    z = hac.linkage(X, method='single')

    # Plotting
    knee = np.diff(z[::-1, 2], 2)
    print(knee)

    num_clust1 = knee.argmax() + 2
    knee[knee.argmax()] = 0
    num_clust2 = knee.argmax() + 2

    part1 = hac.fcluster(z, num_clust1, 'maxclust')
    part2 = hac.fcluster(z, num_clust2, 'maxclust')
    print(part1)
    return part1

    # clr = ['#2200CC' ,'#D9007E' ,'#FF6600' ,'#FFCC00' ,'#ACE600' ,'#0099CC' ,
    # '#8900CC' ,'#FF0000' ,'#FF9900' ,'#FFFF00' ,'#00CC01' ,'#0055CC']

    # for part, ax in zip([part1, part2], axes[1:]):
    #     for cluster in set(part):
    #         ax.scatter(a[part == cluster, 0], a[part == cluster, 1], 
    #                    color=clr[cluster])

    # m = '\n(method: {})'.format(method)
    # plt.setp(axes[0], title='Screeplot{}'.format(m), xlabel='partition',
    #          ylabel='{}\ncluster distance'.format(m))
    # plt.setp(axes[1], title='{} Clusters'.format(num_clust1))
    # plt.setp(axes[2], title='{} Clusters'.format(num_clust2))

script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/SMD/')

use_data = 'machine-1-6'
data,_,_,_,_,_ = load_SMD(data_path, use_data)

d = data[:,0].reshape(-1,1)[::10]
# print(d.shape)
# label = hierarchal_clustering(d)
label = hierarchal_clustering2(d)
plt.plot(d)
plt.plot(label)
plt.savefig('fc.png')