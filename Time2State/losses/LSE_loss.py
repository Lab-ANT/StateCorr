import torch
import numpy
import math
import numpy as np
import torch

def hanning_tensor(X):
    length = X.size(2)
    weight = (1-np.cos(2*math.pi*np.arange(length)/length))/2
    weight = torch.tensor(weight)
    return weight.cuda()*X

# def hanning_tensor(X):
#     length = X.shape[2]
#     half_len = int(length/2)
#     quarter_len = int(length/4)
#     margin_weight = (1-np.cos(2*math.pi*np.arange(half_len)/half_len))/2
#     weight = np.ones((length,))
#     weight[:quarter_len] = margin_weight[:quarter_len]
#     weight[3*quarter_len:] = margin_weight[quarter_len:]
#     weight = torch.tensor(weight)
#     return weight.cuda()*X

class LSELoss(torch.nn.modules.loss._Loss):
    """
    Triplet loss for representations of time series. Optimized for training
    sets where all time series have the same length.

    Takes as input a tensor as the chosen batch to compute the loss,
    a PyTorch module as the encoder, a 3D tensor (`B`, `C`, `L`) containing
    the training set, where `B` is the batch size, `C` is the number of
    channels and `L` is the length of the time series, as well as a boolean
    which, if True, enables to save GPU memory by propagating gradients after
    each loss term, instead of doing it after computing the whole loss.

    The triplets are chosen in the following manner. First the size of the
    positive and negative samples are randomly chosen in the range of lengths
    of time series in the dataset. The size of the anchor time series is
    randomly chosen with the same length upper bound but the the length of the
    positive samples as lower bound. An anchor of this length is then chosen
    randomly in the given time series of the train set, and positive samples
    are randomly chosen among subseries of the anchor. Finally, negative
    samples of the chosen length are randomly chosen in random time series of
    the train set.

    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty, M, N):
        super(LSELoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty
        self.M = M
        self.N = N
        # temperature parameter
        self.tau = 1
        # self.lambda1 = 1

    def forward(self, batch, encoder, train, save_memory=False):
        # length_pos_neg = numpy.random.randint(1, high=length + 1)
        M = self.M
        N = self.N
        length_pos_neg=self.compared_length
        
        total_length = batch.size(2)
        # multiplicative_ratio = self.negative_penalty / N
        center_list = []
        loss1 = 0
        for i in range(M):
            random_pos = numpy.random.randint(0, high=total_length - length_pos_neg*2 + 1, size=self.nb_random_samples)
            rand_samples = [batch[0,:, i: i+length_pos_neg] for i in range(random_pos[0],random_pos[0]+N)]
            # print(random_pos)
            embeddings = encoder(hanning_tensor(torch.stack(rand_samples)))
            # embeddings = encoder(torch.stack(rand_samples))
            # print(embeddings.shape)
            size_representation = embeddings.size(1)

            for i in range(N):
                for j in range(N):
                    if j<=i:
                        continue
                    else:
                        loss1 += -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                            embeddings[i].view(1, 1, size_representation),
                            embeddings[j].view(1, size_representation, 1))/self.tau))
            center = torch.mean(embeddings, dim=0)
            center_list.append(center)
        
        loss2=0
        for i in range(M):
            for j in range(M):
                if j<=i:
                    continue
                loss2 += -torch.mean(torch.nn.functional.logsigmoid(-torch.bmm(
                    center_list[i].view(1, 1, size_representation),
                    center_list[j].view(1, size_representation, 1))/self.tau))

        loss = loss1/(M*N*(N-1)/2) + loss2/(M*(M-1)/2)
        # loss = loss2/(M*(M-1)/2)
        return loss