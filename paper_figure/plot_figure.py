import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import matplotlib

# ==========UCR-SEG=========
# UCR-SEG Time2State 0.4365556516872703 0.4437049697471076
# UCR-SEG TICC 0.1981515729753073 0.21693679987083414
# UCR-SEG HDP_HSMM 0.15039113682690508 0.2578517423636945
# UCR-SEG AutoPlait 0.06953938204046246 0.10323196338607293
# UCR-SEG ClaSP 0.4467451723924388 0.5034622463639125
# UCR-SEG HVGH 0.06532986347995076 0.1439801604759431
# -------- ARI --------
# Time2State ['ClaSP']
# TICC ['HDP_HSMM']
# HDP_HSMM ['TICC']
# AutoPlait ['HVGH']
# ClaSP ['Time2State']
# HVGH ['AutoPlait']
# -------- NMI --------
# Time2State ['ClaSP']
# TICC ['HDP_HSMM', 'AutoPlait', 'HVGH']
# HDP_HSMM ['TICC']
# AutoPlait ['TICC', 'HVGH']
# ClaSP ['Time2State']
# HVGH ['TICC', 'AutoPlait']
# ==========ActRecTut=========
# -------- ARI --------
# Time2State ['TICC']
# TICC ['Time2State', 'HDP_HSMM', 'ClaSP']
# HDP_HSMM ['TICC', 'ClaSP']
# AutoPlait ['ClaSP', 'HVGH']
# ClaSP ['TICC', 'HDP_HSMM', 'AutoPlait', 'HVGH']
# HVGH ['AutoPlait', 'ClaSP']
# -------- NMI --------
# Time2State ['TICC']
# TICC ['Time2State', 'HDP_HSMM', 'ClaSP']
# HDP_HSMM ['TICC', 'ClaSP']
# AutoPlait ['ClaSP']
# ClaSP ['TICC', 'HDP_HSMM', 'AutoPlait', 'HVGH']
# HVGH ['ClaSP']
# ==========PAMAP2=========
# -------- ARI --------
# Time2State ['TICC', 'HDP_HSMM', 'ClaSP']
# TICC ['Time2State', 'HDP_HSMM', 'ClaSP']
# HDP_HSMM ['Time2State', 'TICC', 'ClaSP']
# AutoPlait ['HVGH']
# ClaSP ['Time2State', 'TICC', 'HDP_HSMM']
# HVGH ['AutoPlait']
# -------- NMI --------
# Time2State ['TICC', 'ClaSP']
# TICC ['Time2State', 'ClaSP']
# HDP_HSMM []
# AutoPlait []
# ClaSP ['Time2State', 'TICC']
# HVGH []

# def ARI():
#     plt.figure(figsize=(10, 3))
#     labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'UCR-SEG']
#     Time2State =  [0.8125, 0.7507, 0.7670, 0.3135, 0.6522, 0.4320]
#     TICC_      =  [0.6242, 0.7255, 0.7839, 0.3008, 0.3947, 0.2331]
#     Autoplait_ =  [0.0713, 0.8057, 0.0586, 0.0001, 0.2948, 0.0695]
#     HVGH_      =  [0.0809, 0.0500, 0.0881, 0.0032, 0.0788, 0.0653]
#     HDP_HSMM   =  [0.6462, 0.6137, 0.6644, 0.2882, 0.4678, 0.1648]
#     ClaSP      =  [0.2941, 0.5790, 0.1722, 0.3269, 0.6023, 0.5042]

#     x = np.arange(len(labels))  # pos of x-ticks.
#     width = 0.14  # width of bar
#     plt.style.use('ggplot')
#     plt.bar(x - 2.5*width, Time2State, width, label='Time2State',hatch='//.')
#     plt.bar(x - 1.5*width, TICC_, width, label='TICC', hatch='/.')
#     plt.bar(x - .5*width, HDP_HSMM, width, label='HDP-HSMM', hatch='\\.')
#     plt.bar(x + .5*width, Autoplait_, width, label='AutoPlait', hatch='///.')
#     plt.bar(x + 1.5*width, HVGH_, width, label='HVGH', hatch='\\\\\\.')
#     plt.bar(x + 2.5*width, ClaSP, width, label='ClaSP', hatch='\\\\\\.')
#     plt.scatter([x[3]+0.5*width],[0.05], marker='x', label='refuse to work', color='red')

    
#     synthetic = ['a','c','b','f','e','d']
#     MoCap = ['a','ab','b','a','c','b']
#     USC_HAD = ['a','c','b','d','e','a']
#     ActRecTut = ['a','a','a','b','b','ab']
#     PAMAP2 = ['a','a','a','','b','a']
#     UCR_SEG = ['a','b','b','c','c','a']
#     significant_list = [synthetic,MoCap,ActRecTut,PAMAP2,USC_HAD,UCR_SEG]
#     mehtod_list = [Time2State, TICC_, HDP_HSMM, Autoplait_, HVGH_, ClaSP]
#     pos = -2.5
#     for dataset_num in range(6):
#         for i in range(6):
#             plt.text((x[dataset_num]+pos*width), mehtod_list[i][dataset_num], significant_list[dataset_num][i], ha='center', va='bottom', color="k", fontsize=12)
#             # plt.text((x[dataset_num]+pos*width), mehtod_list[0][0], significant_list[dataset_num][i], ha='center', va='bottom', color="k", fontsize=12)
#             pos+=1
#         pos=-2.5

#     plt.ylabel('ARI', size=15)
#     plt.xticks(x, labels=labels, size=15)
#     plt.yticks(size=15)
#     plt.legend(bbox_to_anchor=(0.5, 1.2), ncol=7, fontsize=10, loc='upper center')
#     plt.tight_layout()
#     plt.show()

# def NMI():
#     plt.figure(figsize=(10, 3))
#     # labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD']
#     labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'UCR-SEG']
#     Time2State  =  [0.8407, 0.7670, 0.7170, 0.5509, 0.8028, 0.4437]
#     TICC_ =       [0.7489, 0.7524, 0.7466, 0.5262, 0.7028, 0.2169]
#     # GHMM       =  [0.4661, 0.4816, 0.6456, 0.4001, 0.5524]
#     Autoplait_ =  [0.1307, 0.8289, 0.1418, 0.0000, 0.5413, 0.1032]
#     HVGH_ =       [0.1606, 0.1523, 0.2088, 0.0374, 0.1883, 0.1439]
#     HDP_HSMM   =  [0.8105, 0.7168, 0.6796, 0.5477, 0.7219, 0.2578]
#     ClaSP      =  [0.3229, 0.5790, 0.1722, 0.3269, 0.6933, 0.5034]

#     x = np.arange(len(labels))  # pos of x-ticks.
#     width = 0.14  # width of bar
#     plt.style.use('ggplot')
#     plt.bar(x - 2.5*width, Time2State, width, label='Time2State',hatch='//.')
#     plt.bar(x - 1.5*width, TICC_, width, label='TICC', hatch='/.')
#     plt.bar(x - .5*width, HDP_HSMM, width, label='HDP-HSMM', hatch='///.')
#     plt.bar(x + .5*width, Autoplait_, width, label='AutoPlait', hatch='\\\\\\.')
#     plt.bar(x + 1.5*width, HVGH_, width, label='HVGH', hatch='\\.')
#     plt.bar(x + 2.5*width, ClaSP, width, label='ClaSP', hatch='\\\\.')

#     # plt.scatter([3.14],[0.05], marker='x', label='refuse\nto work', color='red')
#     plt.scatter([x[3]+0.5*width],[0.05], marker='x', label='refuse to work', color='red')

#     synthetic = ['a','c','b','f','e','d']
#     MoCap = ['a','a','ab','a','c','b']
#     USC_HAD = ['a','c','b','d','e','a']
#     ActRecTut = ['a','a','a','b','b','ab']
#     PAMAP2 = ['a','a','a','','b','a']
#     UCR_SEG = ['a','bc','b','c','c','a']
#     significant_list = [synthetic,MoCap,ActRecTut,PAMAP2,USC_HAD,UCR_SEG]
#     mehtod_list = [Time2State, TICC_, HDP_HSMM, Autoplait_, HVGH_, ClaSP]
#     pos = -2.5
#     for dataset_num in range(6):
#         for i in range(6):
#             plt.text((x[dataset_num]+pos*width), mehtod_list[i][dataset_num], significant_list[dataset_num][i], ha='center', va='bottom', color="k", fontsize=12)
#             # plt.text((x[dataset_num]+pos*width), mehtod_list[0][0], significant_list[dataset_num][i], ha='center', va='bottom', color="k", fontsize=12)
#             pos+=1
#         pos=-2.5

#     plt.ylabel('NMI', size=15)
#     plt.xticks(x, labels=labels, size=15)
#     plt.yticks(size=15)
#     plt.legend(bbox_to_anchor=(0.5, 1.2), ncol=7, fontsize=10, loc='upper center')
#     plt.tight_layout()
#     plt.show()

def ARI():
    plt.figure(figsize=(10, 3))
    labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD']
    Time2State =  [0.8125, 0.7507, 0.7691, 0.3236, 0.6522]
    TICC_      =  [0.6242, 0.7255, 0.7711, 0.3060, 0.3947]
    Autoplait_ =  [0.0713, 0.8057, 0.0586, 0.0001, 0.2948]
    HVGH_      =  [0.0809, 0.0500, 0.0881, 0.0032, 0.0788]
    HDP_HSMM   =  [0.6462, 0.6137, 0.5396, 0.2723, 0.4448]
    ClaSP      =  [0.2941, 0.5453, 0.1722, 0.2814, 0.5064]

    x = np.arange(len(labels))  # pos of x-ticks.
    width = 0.14  # width of bar
    plt.style.use('ggplot')
    plt.bar(x - 2.5*width, Time2State, width, label='Time2State',hatch='//.', yerr=[0.05 for i in range(5)])
    plt.bar(x - 1.5*width, TICC_, width, label='TICC', hatch='/.')
    plt.bar(x - .5*width, HDP_HSMM, width, label='HDP-HSMM', hatch='\\.')
    plt.bar(x + .5*width, ClaSP, width, label='ClaSP', hatch='///.')
    plt.bar(x + 1.5*width, Autoplait_, width, label='AutoPlait', hatch='\\\\\\.')
    plt.bar(x + 2.5*width, HVGH_, width, label='HVGH', hatch='\\\\\\.')
    plt.scatter([x[3]+1.5*width],[0.05], marker='x', label='refuse to work', color='red')

    # synthetic = ['a','c','b','d','f','e']
    # MoCap = ['a','ab','b','b','a','c']
    # USC_HAD = ['a','d','c','b','e','f']
    # ActRecTut = ['a','a','a','ab','b','b']
    # PAMAP2 = ['a','a','a','a','','b']
    # UCR_SEG = ['a','b','b','a','c','c']
    # significant_list = [synthetic,MoCap,ActRecTut,PAMAP2,USC_HAD,UCR_SEG]
    # mehtod_list = [Time2State, TICC_, HDP_HSMM, ClaSP, Autoplait_, HVGH_]
    # pos = -2.5
    # for dataset_num in range(6):
    #     for i in range(6):
    #         plt.text((x[dataset_num]+pos*width), mehtod_list[i][dataset_num], significant_list[dataset_num][i], ha='center', va='bottom', color="k", fontsize=12)
    #         # plt.text((x[dataset_num]+pos*width), mehtod_list[0][0], significant_list[dataset_num][i], ha='center', va='bottom', color="k", fontsize=12)
    #         pos+=1
    #     pos=-2.5

    plt.ylabel('ARI', size=15)
    plt.xticks(x, labels=labels, size=15)
    plt.yticks(size=15)
    plt.legend(bbox_to_anchor=(0.5, 1.2), ncol=7, fontsize=10, loc='upper center')
    plt.tight_layout()
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.show()

def NMI():
    plt.figure(figsize=(10, 3))
    # labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD']
    labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'UCR-SEG']
    Time2State  =  [0.8004, 0.7564, 0.7400, 0.5911, 0.8123, 0.4437]
    TICC_ =       [0.7489, 0.7497, 0.7416, 0.5942, 0.7028, 0.2169]
    Autoplait_ =  [0.1307, 0.8289, 0.1418, 0.0000, 0.5413, 0.1032]
    HVGH_ =       [0.1606, 0.1523, 0.2088, 0.0374, 0.1883, 0.1439]
    HDP_HSMM   =  [0.7804, 0.7237, 0.6472, 0.5328, 0.6838, 0.2578]
    ClaSP      =  [0.4484, 0.6773, 0.2308, 0.5820, 0.6933, 0.5034]

    x = np.arange(len(labels))  # pos of x-ticks.
    width = 0.14  # width of bar
    plt.style.use('ggplot')
    plt.bar(x - 2.5*width, Time2State, width, label='Time2State',hatch='//.')
    plt.bar(x - 1.5*width, TICC_, width, label='TICC', hatch='/.')
    plt.bar(x - .5*width, HDP_HSMM, width, label='HDP-HSMM', hatch='///.')
    plt.bar(x + .5*width, ClaSP, width, label='ClaSP', hatch='\\\\\\.')
    plt.bar(x + 1.5*width, Autoplait_, width, label='AutoPlait', hatch='\\.')
    plt.bar(x + 2.5*width, HVGH_, width, label='HVGH', hatch='\\\\.')

    # plt.scatter([3.14],[0.05], marker='x', label='refuse\nto work', color='red')
    plt.scatter([x[3]+1.5*width],[0.05], marker='x', label='refuse to work', color='red')

    synthetic = ['a','b','a','c','d','d']
    MoCap = ['ab','ab','b','b','a','c']
    USC_HAD = ['a','b','b','b','c','d']
    ActRecTut = ['a','a','a','b','b','b']
    PAMAP2 = ['a','a','a','a','','b']
    UCR_SEG = ['a','bc','b','a','c','c']
    significant_list = [synthetic,MoCap,ActRecTut,PAMAP2,USC_HAD,UCR_SEG]
    mehtod_list = [Time2State, TICC_, HDP_HSMM, ClaSP, Autoplait_, HVGH_]
    pos = -2.5
    for dataset_num in range(6):
        for i in range(6):
            plt.text((x[dataset_num]+pos*width), mehtod_list[i][dataset_num], significant_list[dataset_num][i], ha='center', va='bottom', color="k", fontsize=12)
            # plt.text((x[dataset_num]+pos*width), mehtod_list[0][0], significant_list[dataset_num][i], ha='center', va='bottom', color="k", fontsize=12)
            pos+=1
        pos=-2.5

    plt.ylabel('NMI', size=15)
    plt.xticks(x, labels=labels, size=15)
    plt.yticks(size=15)
    plt.legend(bbox_to_anchor=(0.5, 1.2), ncol=7, fontsize=10, loc='upper center')
    plt.tight_layout()
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.show()

ARI()
# NMI()