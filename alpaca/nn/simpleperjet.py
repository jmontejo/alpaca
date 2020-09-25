import torch

from alpaca.nn.feedforward import FeedForwardHead
import numpy as np

__all__ = ['SimpleNNperjet']


class SimpleNNperjet(torch.nn.Module):

    permutations = [ #3 first are lep+met
           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
           [0, 1, 2, 4, 3, 5, 6, 7, 8, 9],
           [0, 1, 2, 4, 5, 3, 6, 7, 8, 9],
           [0, 1, 2, 4, 5, 6, 3, 7, 8, 9],
           [0, 1, 2, 4, 5, 6, 7, 3, 8, 9],
           [0, 1, 2, 4, 5, 6, 7, 8, 3, 9],
           [0, 1, 2, 4, 5, 6, 7, 8, 9, 3],
    ]

    def __init__(self, nobjects, noutputs, fflayers=[200], usebjet=False):
        super(SimpleNNperjet, self).__init__()
        self.njets = nobjects -3
        assert (noutputs%self.njets ==0), "Outputs has to be a multiple of inputs"
        self.usebjet = usebjet
        factor = 5 if usebjet else 4
        self.noutputs = noutputs
        self.ntotal = nobjects
        self.norm = torch.nn.BatchNorm1d(self.ntotal * factor)
        self.head = FeedForwardHead([self.ntotal * factor] + fflayers + [noutputs//self.njets])

    # Jets go in, labels come out
    def forward(self, vectors):

        if not self.usebjet:
            vectors = vectors[:,:,:-1] #drop bjet info

        fulloutput = []
        for perm in self.permutations:
            idx = np.empty_like(perm)
            idx[perm] = np.arange(len(perm))
            tmpvectors = vectors[:,idx]

            output = tmpvectors.reshape(tmpvectors.shape[0], -1)
            output = self.norm(output)
            output = self.head(output)

            fulloutput.append(output)
        output = torch.cat(fulloutput, dim=1)
        output = output.reshape(output.shape[0],self.njets, self.noutputs//self.njets)
        output = output.permute(0,2,1)
        output = output.reshape(output.shape[0],-1)

        return output
