import torch
import numpy as np

from alpaca.nn.colalola import CoLa, LoLa
from alpaca.nn.feedforward import FeedForwardHead

__all__ = ['IsrHead', 'DecayHead', 'Hydra']


class IsrHead(torch.nn.Module):

    def __init__(self, nobjects, ncombos, fflayers=[200], nscalars=0, nextrafields=0):
        super(IsrHead, self).__init__()
        self.nobjects = nobjects
        self.ntotal = nobjects + ncombos
        self.cola = CoLa(nobjects, ncombos)
        self.lola = LoLa(self.ntotal, nextrafields)
        self.norm = torch.nn.BatchNorm1d(self.ntotal * (5+nextrafields)+nscalars)
        # This one does top jet identification
        self.head = FeedForwardHead([self.ntotal * (5+nextrafields)+nscalars] + fflayers + [nobjects])

    def forward(self,vectors, scalars=0):
        output = self.cola(vectors)
        output = self.lola(output)
        output = output.reshape(output.shape[0], -1)
        if scalars:
            output = torch.cat([scalars,output],1)
        output = self.norm(output)
        output = self.head(output)
        return output


class DecayHead(torch.nn.Module):

    def __init__(self, nobjects, ncombos, fflayers=[200], nscalars=0, nextrafields=0):
        super(DecayHead, self).__init__()
        self.nobjects = nobjects
        self.ntotal = 6 + ncombos
        self.cola = CoLa(6, ncombos)
        self.lola = LoLa(self.ntotal, nextrafields)
        self.norm = torch.nn.BatchNorm1d(self.ntotal * (5+nextrafields)+nscalars)
        # This one reconstructs the decay process, taking the ISR result as input
        # Both the top decay matching and the b-tagging
        # (these seem like things that should emerge simultaneously)
        self.head = FeedForwardHead([self.ntotal * (5+nextrafields)+nscalars] + fflayers + [11])

    def forward(self,selected_vector, scalars=0):
        output = self.cola(selected_vector)
        output = self.lola(output)
        output = output.reshape(output.shape[0], -1)
        if scalars: 
            output = torch.cat([scalars,output],1)
        output = self.norm(output)
        output = self.head(output)
        return output


# A NN with multiple output layers & heads
# Technically now the components are no longer just separate heads... oops
class Hydra(torch.nn.Module):

    def __init__(self, nobjects, ncombos, fflayers=[200], nscalars=0, nextrafields=0):
        super(Hydra, self).__init__()
        self.nobjects = nobjects
        self.ncombos = ncombos
        self.fflayers = fflayers
        self.nscalars = nscalars
        self.nextrafields = nextrafields
        self.isr_head = IsrHead(self.nobjects,self.ncombos,self.fflayers, self.nscalars, self.nextrafields)
        self.decay_head = DecayHead(self.nobjects, self.ncombos,self.fflayers, self.nscalars, self.nextrafields)

    def forward(self,vectors):
        if self.nscalars:
            scalars = vectors[:,:self.nscalars]
            vectors = vectors[:,self.nscalars:]            
        else: scalars = 0

        try:
            vectors = vectors.reshape(vectors.shape[0],self.nobjects,4+self.nextrafields)
        except RuntimeError as e:
            print("Hydra objects %d, jet components %d, scalars %d"%(self.nobjects,4+self.nextrafields,self.nscalars ))
            raise e

        # Get the ISR tag result
        output_isr = self.isr_head(vectors, scalars)

        # Extract the best 6 jets
        isrtag = output_isr.detach().numpy()
        selected_vectors = vectors.numpy()
        for ij in range(self.nobjects-6):
            minindices = isrtag.argmin(1)
            isrtag = np.array([np.delete(r,minindices[ie],0) for ie,r in enumerate(isrtag)])
            selected_vectors = np.array([np.delete(r,minindices[ie],0) for ie,r in enumerate(selected_vectors)])
        isrtag = torch.from_numpy(isrtag)
        selected_vectors = torch.from_numpy(selected_vectors)
        output_decay = self.decay_head(selected_vectors)
        # Concatenate the outputs because this gives the same return structure
        # as single-headed networks
        # We can still split up the output later
        return torch.cat([output_isr,output_decay],1)
