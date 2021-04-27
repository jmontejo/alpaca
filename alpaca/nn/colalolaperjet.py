import torch

from alpaca.nn.feedforward import FeedForwardHead
from alpaca.nn.colalola import CoLa, LoLa
import numpy as np

__all__ = ['CoLaLoLaPerJet']


class CoLaLoLaPerJet(torch.nn.Module):


    def __init__(self, njets, nextras, ncombos, noutputs, nscalars=0, nextrafields=0, fflayers=[200]):
        super(CoLaLoLaPerJet, self).__init__()
        assert njets+nextras > 0, "No sense in using CoLaLoLa with only flat variables"
        self.njets = njets
        self.nextras = nextras
        self.nobjects = njets+nextras
        assert (noutputs%self.njets ==0), "Outputs has to be a multiple of inputs"
        self.noutputs = noutputs
        self.ntotal = self.nobjects + ncombos
        self.cola = CoLa(self.nobjects, ncombos)
        self.lola = LoLa(self.ntotal, nextrafields)
        factor = (5+nextrafields) 
        self.norm = torch.nn.BatchNorm1d(self.ntotal * factor +nscalars)
        self.head = FeedForwardHead([self.ntotal * factor+nscalars] + fflayers + [noutputs//self.njets])
        self.nscalars = nscalars
        self.nextrafields = nextrafields
        self.permutations = [ list(range(nextras))+[nextras if i==perm else p+(i<perm) for i,p in enumerate(range(nextras,nextras+njets))] for perm in range(njets)]

    '''
    The permutation formula generates something like this
    permutations = [ #3 first are extras, e.g. lep+met
           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
           [0, 1, 2, 4, 3, 5, 6, 7, 8, 9],
           [0, 1, 2, 4, 5, 3, 6, 7, 8, 9],
           [0, 1, 2, 4, 5, 6, 3, 7, 8, 9],
           [0, 1, 2, 4, 5, 6, 7, 3, 8, 9],
           [0, 1, 2, 4, 5, 6, 7, 8, 3, 9],
           [0, 1, 2, 4, 5, 6, 7, 8, 9, 3],
    ]
    '''


    def forward(self, vectors):
        #truncate vectors in 4-vectors + scalars, feed jets and merge with scalars after LoLa
        if self.nscalars:
            scalars = vectors[:,-self.nscalars:]
            vectors = vectors[:,:-self.nscalars]
        try:
            vectors = vectors.reshape(vectors.shape[0],self.nobjects,4+self.nextrafields)
        except RuntimeError as e:
            print("ColaLola objects %d, objects+combos %d, jet components %d, scalars %d"%(self.nobjects,self.ntotal,4+self.nextrafields,self.nscalars ))
            print(vectors.reshape(vectors.shape[0],30))
            raise e

        fulloutput = []
        for perm in self.permutations:
            idx = np.empty_like(perm)
            idx[perm] = np.arange(len(perm))
            tmpvectors = vectors[:,idx]

            output = self.cola(tmpvectors)
            output = self.lola(output)
            output = output.reshape(output.shape[0], -1)
            if self.nscalars:
                output = torch.cat([scalars,output],1)
            output = self.norm(output)
            output = self.head(output)

            fulloutput.append(output)
        output = torch.cat(fulloutput, dim=1)
        output = output.reshape(output.shape[0],self.njets, self.noutputs//self.njets)
        output = output.permute(0,2,1)
        output = output.reshape(output.shape[0],-1)

        return output