import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# def ARI():
#     plt.figure(figsize=(10, 3))
#     labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'UCR-SEG']
#     LSE      = [0.8125, 0.7507, 0.7691, 0.3236, 0.6503, 0.4437]
#     Triplet  = [0.6853, 0.5953, 0.6863, 0.2662, 0.5556, 0.3115]
#     TNC      = [0.6407, 0.6296, 0.6011, 0.2956, 0.4994, 0.2378]
#     CPC      = [0.4870, 0.5708, 0.5286, 0.2402, 0.5033, 0.2090]
#     TS2Vec   = [0.7780, 0.5157, 0.6685, 0.2716, 0.5039, 0.2463]

#     x = np.arange(len(labels))  # pos of x-ticks.
#     width = 0.14  # width of bar
#     plt.style.use('ggplot')

#     plt.bar(x - 2*width, LSE, width, label='LSE',hatch='//.')
#     plt.bar(x - 1*width, Triplet, width, label='Triplet',hatch='\\\\.')
#     plt.bar(x + 0*width, TNC, width, label='TNC', hatch='/.')
#     plt.bar(x + 1*width, CPC, width, label='CPC', hatch='\\.')
#     plt.bar(x + 2*width, TS2Vec, width, label='TS2Vec', hatch='\\.')

#     synthetic = ['a','c','d','e','b']
#     MoCap = ['a','ab','ab','ab','b']
#     USC_HAD = ['a','b','c','c','c']
#     ActRecTut = ['a','b','c','c','b']
#     PAMAP2 = ['a','ab','ab','b','ab']
#     UCR_SEG = ['a','ab','b','b','b']
#     significant_list = [synthetic,MoCap,ActRecTut,PAMAP2,USC_HAD,UCR_SEG]
#     mehtod_list = [LSE, Triplet, TNC, CPC, TS2Vec]
#     pos = -2
#     for dataset_num in range(6):
#         for i in range(5):
#             plt.text((x[dataset_num]+pos*width),
#             mehtod_list[i][dataset_num],
#             significant_list[dataset_num][i], ha='center', va='bottom', color="k", fontsize=10)
#             pos+=1
#         pos=-2

#     plt.ylabel('ARI', size=15)
#     plt.xticks(x, labels=labels, size=15)
#     plt.yticks(size=15)
#     plt.legend(bbox_to_anchor=(0.5, 1.25), ncol=5, fontsize=15, loc='upper center')
#     plt.tight_layout()
#     plt.show()

# def NMI():
#     plt.figure(figsize=(10, 3))
#     labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'UCR-SEG']
#     LSE      = [0.8004, 0.7564, 0.7400, 0.5911, 0.8123, 0.4437]
#     Triplet  = [0.7503, 0.6789, 0.6622, 0.5083, 0.7396, 0.3394]
#     TNC      = [0.7316, 0.6868, 0.6104, 0.5360, 0.6965, 0.2648]
#     CPC      = [0.6417, 0.5973, 0.5644, 0.4699, 0.6559, 0.2352]
#     TS2Vec   = [0.7836, 0.6294, 0.6593, 0.4973, 0.7086, 0.2642]

#     x = np.arange(len(labels))  # pos of x-ticks.
#     width = 0.14  # width of bar
#     plt.style.use('ggplot')
#     plt.bar(x - 2*width, LSE, width, label='LSE',hatch='//.')
#     plt.bar(x - 1*width, Triplet, width, label='Triplet',hatch='\\\\.')
#     plt.bar(x + 0*width, TNC, width, label='TNC', hatch='/.')
#     plt.bar(x + 1*width, CPC, width, label='CPC', hatch='\\.')
#     plt.bar(x + 2*width, TS2Vec, width, label='TS2Vec', hatch='\\.')

#     synthetic = ['a','c','b','f','a']
#     MoCap = ['a','ab','ab','ab','b']
#     USC_HAD = ['a','b','c','d','c']
#     ActRecTut = ['a','b','c','c','b']
#     PAMAP2 = ['a','b','bc','bc','bc']
#     UCR_SEG = ['a','ab','b','b','b']
#     significant_list = [synthetic,MoCap,ActRecTut,PAMAP2,USC_HAD,UCR_SEG]
#     mehtod_list = [LSE, Triplet, TNC, CPC, TS2Vec]
#     pos = -2
#     for dataset_num in range(6):
#         for i in range(5):
#             plt.text((x[dataset_num]+pos*width),
#             mehtod_list[i][dataset_num],
#             significant_list[dataset_num][i], ha='center', va='bottom', color="k", fontsize=11)
#             pos+=1
#         pos=-2

#     plt.ylabel('NMI', size=15)
#     plt.xticks(x, labels=labels, size=15)
#     plt.yticks(size=15)
#     plt.legend(bbox_to_anchor=(0.5, 1.25), ncol=5, fontsize=15, loc='upper center')
#     plt.tight_layout()
#     plt.show()

# def ARI():
#     plt.figure(figsize=(10, 3.5))
#     labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'UCR-SEG']
#     LSE      = [0.8125, 0.7507, 0.7691, 0.3236, 0.6503, 0.4437]
#     Triplet  = [0.6853, 0.5953, 0.6863, 0.2662, 0.5556, 0.3115]
#     TNC      = [0.6407, 0.6296, 0.6011, 0.2956, 0.4994, 0.2378]
#     CPC      = [0.4870, 0.5708, 0.5286, 0.2402, 0.5033, 0.2090]
#     TS2Vec   = [0.7780, 0.5157, 0.6685, 0.2716, 0.5039, 0.2463]

#     x = np.arange(len(labels))  # pos of x-ticks.
#     width = 0.14  # width of bar
#     plt.style.use('ggplot')

#     plt.bar(x - 2*width, LSE, width, label='LSE',hatch='//.')
#     plt.bar(x - 1*width, TS2Vec, width, label='TS2Vec',hatch='\\\\.')
#     plt.bar(x + 0*width, Triplet, width, label='Triplet', hatch='/.')
#     plt.bar(x + 1*width, TNC, width, label='TNC', hatch='\\.')
#     plt.bar(x + 2*width, CPC, width, label='CPC', hatch='\\.')

#     synthetic = ['a','b','c','d','e']
#     MoCap = ['a','b','ab','ab','ab']
#     USC_HAD = ['a','c','b','c','c']
#     ActRecTut = ['a','b','b','c','c']
#     PAMAP2 = ['a','ab','ab','ab','b']
#     # UCR_SEG = ['a','b','ab','b','b']
#     # significant_list = [synthetic,MoCap,ActRecTut,PAMAP2,USC_HAD,UCR_SEG]
#     significant_list = [synthetic,MoCap,ActRecTut,PAMAP2,USC_HAD]
#     mehtod_list = [LSE, TS2Vec, Triplet, TNC, CPC]
#     pos = -2
#     for dataset_num in range(5):
#         for i in range(5):
#             plt.text((x[dataset_num]+pos*width),
#             mehtod_list[i][dataset_num],
#             significant_list[dataset_num][i], ha='center', va='bottom', color="k", fontsize=10)
#             pos+=1
#         pos=-2

#     plt.ylabel('ARI', size=15)
#     plt.xticks(x, labels=labels, size=15)
#     plt.yticks(size=15)
#     plt.legend(bbox_to_anchor=(0.5, 1.25), ncol=5, fontsize=15, loc='upper center')
#     plt.tight_layout()
#     plt.show()

# def NMI():
#     plt.figure(figsize=(10, 3.5))
#     labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD', 'UCR-SEG']
#     LSE      = [0.8004, 0.7564, 0.7400, 0.5911, 0.8123, 0.4437]
#     Triplet  = [0.7503, 0.6789, 0.6622, 0.5083, 0.7396, 0.3394]
#     TNC      = [0.7316, 0.6868, 0.6104, 0.5360, 0.6965, 0.2648]
#     CPC      = [0.6417, 0.5973, 0.5644, 0.4699, 0.6559, 0.2352]
#     TS2Vec   = [0.7836, 0.6294, 0.6593, 0.4973, 0.7086, 0.2642]

#     x = np.arange(len(labels))  # pos of x-ticks.
#     width = 0.14  # width of bar
#     plt.style.use('ggplot')
#     plt.bar(x - 2*width, LSE, width, label='LSE',hatch='//.')
#     plt.bar(x - 1*width, TS2Vec, width, label='TS2Vec',hatch='\\\\.')
#     plt.bar(x + 0*width, Triplet, width, label='Triplet', hatch='/.')
#     plt.bar(x + 1*width, TNC, width, label='TNC', hatch='\\.')
#     plt.bar(x + 2*width, CPC, width, label='CPC', hatch='\\.')

#     synthetic = ['a','a','b','c','d']
#     MoCap = ['a','b','ab','ab','ab']
#     USC_HAD = ['a','c','b','c','d']
#     ActRecTut = ['a','b','b','c','c']
#     PAMAP2 = ['a','b','b','b','b']
#     # UCR_SEG = ['a','b','ab','b','b']
#     # significant_list = [synthetic,MoCap,ActRecTut,PAMAP2,USC_HAD,UCR_SEG]
#     significant_list = [synthetic,MoCap,ActRecTut,PAMAP2,USC_HAD]
#     mehtod_list = [LSE, TS2Vec, Triplet, TNC, CPC]
#     pos = -2
#     for dataset_num in range(6):
#         for i in range(5):
#             plt.text((x[dataset_num]+pos*width),
#             mehtod_list[i][dataset_num],
#             significant_list[dataset_num][i], ha='center', va='bottom', color="k", fontsize=11)
#             pos+=1
#         pos=-2

#     plt.ylabel('NMI', size=15)
#     plt.xticks(x, labels=labels, size=15)
#     plt.yticks(size=15)
#     plt.legend(bbox_to_anchor=(0.5, 1.25), ncol=5, fontsize=15, loc='upper center')
#     plt.tight_layout()
#     plt.show()

# ARI()
# NMI()

def ARI():
    plt.figure(figsize=(10, 3.5))
    labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD']
    LSE      = [0.8125, 0.7507, 0.7691, 0.3236, 0.6503]
    Triplet  = [0.6853, 0.5953, 0.6863, 0.2662, 0.5556]
    TNC      = [0.6407, 0.6296, 0.6011, 0.2956, 0.4994]
    CPC      = [0.4870, 0.5708, 0.5286, 0.2402, 0.5033]
    TS2Vec   = [0.7780, 0.5157, 0.6685, 0.2716, 0.5039]

    x = np.arange(len(labels))  # pos of x-ticks.
    width = 0.14  # width of bar
    plt.style.use('ggplot')

    plt.bar(x - 2*width, LSE, width, label='LSE',hatch='//.')
    plt.bar(x - 1*width, TS2Vec, width, label='TS2Vec',hatch='\\\\.')
    plt.bar(x + 0*width, Triplet, width, label='Triplet', hatch='/.')
    plt.bar(x + 1*width, TNC, width, label='TNC', hatch='\\.')
    plt.bar(x + 2*width, CPC, width, label='CPC', hatch='\\.')

    synthetic = ['a','b','c','d','e']
    MoCap = ['a','b','ab','ab','ab']
    USC_HAD = ['a','c','b','c','c']
    ActRecTut = ['a','b','b','c','c']
    PAMAP2 = ['a','ab','ab','ab','b']
    # UCR_SEG = ['a','b','ab','b','b']
    # significant_list = [synthetic,MoCap,ActRecTut,PAMAP2,USC_HAD,UCR_SEG]
    significant_list = [synthetic,MoCap,ActRecTut,PAMAP2,USC_HAD]
    mehtod_list = [LSE, TS2Vec, Triplet, TNC, CPC]
    pos = -2
    for dataset_num in range(5):
        for i in range(5):
            plt.text((x[dataset_num]+pos*width),
            mehtod_list[i][dataset_num],
            significant_list[dataset_num][i], ha='center', va='bottom', color="k", fontsize=10)
            pos+=1
        pos=-2

    plt.ylabel('ARI', size=15)
    plt.xticks(x, labels=labels, size=15)
    plt.yticks(size=15)
    plt.legend(bbox_to_anchor=(0.5, 1.25), ncol=5, fontsize=15, loc='upper center')
    plt.tight_layout()
    plt.show()

def NMI():
    plt.figure(figsize=(10, 3.5))
    labels = ['Synthetic', 'MoCap', 'ActRecTut', 'PAMAP2', 'USC-HAD']
    LSE      = [0.8004, 0.7564, 0.7400, 0.5911, 0.8123]
    Triplet  = [0.7503, 0.6789, 0.6622, 0.5083, 0.7396]
    TNC      = [0.7316, 0.6868, 0.6104, 0.5360, 0.6965]
    CPC      = [0.6417, 0.5973, 0.5644, 0.4699, 0.6559]
    TS2Vec   = [0.7836, 0.6294, 0.6593, 0.4973, 0.7086]

    x = np.arange(len(labels))  # pos of x-ticks.
    width = 0.14  # width of bar
    plt.style.use('ggplot')
    plt.bar(x - 2*width, LSE, width, label='LSE',hatch='//.')
    plt.bar(x - 1*width, TS2Vec, width, label='TS2Vec',hatch='\\\\.')
    plt.bar(x + 0*width, Triplet, width, label='Triplet', hatch='/.')
    plt.bar(x + 1*width, TNC, width, label='TNC', hatch='\\.')
    plt.bar(x + 2*width, CPC, width, label='CPC', hatch='\\.')

    synthetic = ['a','a','b','c','d']
    MoCap = ['a','b','ab','ab','ab']
    USC_HAD = ['a','c','b','c','d']
    ActRecTut = ['a','b','b','c','c']
    PAMAP2 = ['a','b','b','b','b']
    # UCR_SEG = ['a','b','ab','b','b']
    # significant_list = [synthetic,MoCap,ActRecTut,PAMAP2,USC_HAD,UCR_SEG]
    significant_list = [synthetic,MoCap,ActRecTut,PAMAP2,USC_HAD]
    mehtod_list = [LSE, TS2Vec, Triplet, TNC, CPC]
    pos = -2
    for dataset_num in range(5):
        for i in range(5):
            plt.text((x[dataset_num]+pos*width),
            mehtod_list[i][dataset_num],
            significant_list[dataset_num][i], ha='center', va='bottom', color="k", fontsize=11)
            pos+=1
        pos=-2

    plt.ylabel('NMI', size=15)
    plt.xticks(x, labels=labels, size=15)
    plt.yticks(size=15)
    plt.legend(bbox_to_anchor=(0.5, 1.25), ncol=5, fontsize=15, loc='upper center')
    plt.tight_layout()
    plt.show()

ARI()
NMI()