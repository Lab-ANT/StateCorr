from cmath import isnan, nan
import numpy as np
import os
import matplotlib.pyplot as plt

num = 5
method_list = ['StateCorr', 'ClaSP', 'HDP-HSMM', 'TICC', 'AutoPlait']

script_path = os.path.dirname(__file__)
fig_save_path = os.path.join(script_path, '../output/figs/')

if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)

def plot():
    plt.style.use('classic')
    fig = plt.figure(figsize=(8,4))
    ax = fig.subplots(ncols=2)
    k = 0
    # for dir in ['output', 'output']:
    for i in [2,3]:
        precision_list = []
        recall_list = []
        for method_name in method_list:
            precision = np.load(os.path.join(script_path, '../output/output_'+method_name+'/precision_RQ%d.npy'%(i)))
            recall = np.load(os.path.join(script_path, '../output/output_'+method_name+'/recall_RQ%d.npy'%(i)))
            precision_list.append(precision)
            recall_list.append(recall)
            print(precision.shape)
        for i, method_name in enumerate(method_list):
            # print(i)
            ax[k].plot(precision_list[i], recall_list[i], lw=2, label=method_name)
            f1 = 2*precision_list[i]*recall_list[i]/(precision_list[i]+recall_list[i])
            f1[np.isnan(f1)] = 0
            idx = np.argmax(f1)
            print(f1)
            x = precision_list[i][idx]
            y = recall_list[i][idx]
            ax[k].plot(x, y, 'o')
            ax[k].text(x*1.01, y*1.01, '%.2f'%(f1[idx]), fontsize=12)
            print(x, y)
        k+=1
    ax[0].set_title('Overall')
    ax[0].set_xlabel('Recall',fontsize=14)
    ax[0].set_ylabel('Precision',fontsize=14)
    ax[0].set_xticks([0.2,0.4,0.6,0.8,1.0])
    # ax[0].set_xlim([0.2,1.01])
    # ax[0].set_ylim([0,1.015])
    ax[0].set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax[1].set_title('Partial')
    ax[1].set_xlabel('Recall',fontsize=14)
    ax[1].set_ylabel('Precision',fontsize=14)
    ax[1].set_xticks([0.2,0.4,0.6,0.8,1.0])
    ax[1].set_yticks([0.2,0.4,0.6,0.8,1.0])
    lines, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, framealpha=1, bbox_to_anchor=(0.98, 1), ncol=5, fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.savefig(os.path.join(fig_save_path, 'prc_RQ3.png'))
    plt.close()
    # plt.show()
plot()