import torch
from .colalola import CoLa, LoLa
from .feedforward import FeedForwardHead
import numpy as np

class IsrHead(torch.nn.Module):
    def __init__(self, njets, ncombos, fflayers=[200]):
        super(IsrHead, self).__init__()
        self.ntotal = njets + ncombos
        self.cola = CoLa(njets, ncombos)
        self.lola = LoLa(self.ntotal)
        self.norm = torch.nn.BatchNorm1d(self.ntotal * 5)
        # This one does top jet identification
        self.head = FeedForwardHead([self.ntotal * 5] + fflayers + [njets])

    def forward(self,vectors):
        output = self.cola(vectors)
        output = self.lola(output)
        output = output.reshape(output.shape[0], -1)
        output = self.norm(output)
        output = self.head(output)
        return output

class DecayHead(torch.nn.Module):
    def __init__(self, njets, ncombos, fflayers=[200]):
        super(DecayHead, self).__init__()
        self.ntotal = 6 + ncombos
        self.njets = njets
        self.cola = CoLa(6, ncombos)
        self.lola = LoLa(self.ntotal)
        self.norm = torch.nn.BatchNorm1d(self.ntotal * 5)
        # This one reconstructs the decay process, taking the ISR result as input
        # Both the top decay matching and the b-tagging
        # (these seem like things that should emerge simultaneously)
        self.head = FeedForwardHead([self.ntotal * 5] + fflayers + [11])

    def forward(self,selected_vector):
        output = self.cola(selected_vector)
        output = self.lola(output)
        output = output.reshape(output.shape[0], -1)
        output = self.norm(output)
        output = self.head(output)
        return output

# A NN with multiple output layers & heads
# Technically now the components are no longer just separate heads... oops
class Hydra(torch.nn.Module):
    def __init__(self, njets, ncombos, fflayers=[200]):
        super(Hydra, self).__init__()
        self.njets = njets
        self.ncombos = ncombos
        self.fflayers = fflayers
        self.isr_head = IsrHead(self.njets,self.ncombos,self.fflayers)
        self.decay_head = DecayHead(self.njets, self.ncombos,self.fflayers)

    def forward(self,vectors):
        # Get the ISR tag result
        output_isr = self.isr_head(vectors)

        # Extract the best 6 jets
        isrtag = output_isr.detach().numpy()
        selected_vectors = vectors.numpy()
        for ij in range(self.njets-6):
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
