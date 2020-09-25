import torch

from alpaca.nn.feedforward import FeedForwardHead

__all__ = ['CoLa', 'LoLa', 'FeedForwardHead', 'CoLaLoLa']


class CoLa(torch.nn.Module):

    def __init__(self, nobjects=15, ncombos=20):
        super(CoLa, self).__init__()
        self.nobjects = nobjects
        self.ncombos = ncombos
        self.outputobj = ncombos + nobjects
        self.identity = torch.eye(self.nobjects)
        # Trainable weights for linear combinations
        self.w_combo = torch.nn.Parameter(torch.randn(self.ncombos,
                                                      self.nobjects))

    # Generate linear combinations of four-vectors
    # Passes on the original four-vectors followed by the combinations
    def forward(self, vectors):
        combo = torch.cat([self.identity, self.w_combo], dim=0)
        combvec = torch.einsum('ij,bjk->bik', combo, vectors)
        return combvec


class LoLa(torch.nn.Module):

    def __init__(self, outputobj, usebjet=False):
        super(LoLa, self).__init__()
        self.outputobj = outputobj
        self.usebjet = usebjet
        self.w_dist = torch.nn.Parameter(torch.randn(self.outputobj,
                                                     self.outputobj))
        self.w_ener = torch.nn.Parameter(torch.randn(self.outputobj,
                                                     self.outputobj))
        self.w_pid  = torch.nn.Parameter(torch.randn(self.outputobj,
                                                    self.outputobj))
        self.w_dl1r = torch.nn.Parameter(torch.randn(self.outputobj,
                                                    self.outputobj))
        self.metric = torch.diag(torch.tensor([-1., -1., -1., 1.]))

    # Calculate Lorentz invariants from the input four-vectors
    # These four-vectors are either the original jets, or the
    # corresponding combinations
    def forward(self, combvec):
        weighted_e = torch.einsum('ij,bj->bi', self.w_ener,combvec[:, :, 0]) #linear combination of energies?
        weighted_p = torch.einsum('ij,bj->bi', self.w_pid,combvec[:, :, 3]) #linear combination of pz?

        if self.usebjet:
            weighted_dl1r = torch.einsum('ij,bj->bi', self.w_dl1r,combvec[:, :, 4]) #linear combination of dl1r


        a = combvec[..., :4].unsqueeze(2).repeat(1, 1, self.outputobj, 1)
        b = combvec[..., :4].unsqueeze(1).repeat(1, self.outputobj, 1, 1)
        diff = (a - b)

        distances = torch.einsum('bnmi,ij,bnmj->bnm', diff, self.metric, diff)
        weighted_d = torch.einsum('nm,bnm->bn', self.w_dist, distances)
        masses = torch.einsum('bni,ij,bnj->bn', combvec[..., :4], self.metric,
                              combvec[..., :4])
        ptsq = combvec[:, :, 1]**2 + combvec[:, :, 2]**2
        if self.usebjet:
            outputs = torch.stack([
                masses,
                ptsq,
                weighted_e,
                weighted_d,
                weighted_p,
                weighted_dl1r,
            ], dim=-1)
        else:
            outputs = torch.stack([
                masses,
                ptsq,
                weighted_e,
                weighted_d,
                weighted_p,
                #weighted_dl1r,
            ], dim=-1)
        return outputs


class CoLaLoLa(torch.nn.Module):

    def __init__(self, nobjects, ncombos, noutputs, fflayers=[200], usebjet=False):
        super(CoLaLoLa, self).__init__()
        self.ntotal = nobjects + ncombos
        self.usebjet = usebjet
        self.cola = CoLa(nobjects, ncombos)
        self.lola = LoLa(self.ntotal, usebjet=usebjet)
        factor = 6 if usebjet else 5
        self.norm = torch.nn.BatchNorm1d(self.ntotal * factor )
        self.head = FeedForwardHead([self.ntotal * factor] + fflayers + [noutputs])

    def forward(self, vectors):
        output = self.cola(vectors)
        output = self.lola(output)
        output = output.reshape(output.shape[0], -1)
        output = self.norm(output)
        output = self.head(output)
        return output
