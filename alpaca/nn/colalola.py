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

    def __init__(self, outputobj, nextrafields=0):
        super(LoLa, self).__init__()
        self.outputobj = outputobj
        self.w_dist = torch.nn.Parameter(torch.randn(self.outputobj,
                                                     self.outputobj))
        self.w_ener = torch.nn.Parameter(torch.randn(self.outputobj,
                                                     self.outputobj))
        self.w_pid = torch.nn.Parameter(torch.randn(self.outputobj,
                                                    self.outputobj))
        self.metric = torch.diag(torch.tensor([-1., -1., -1., 1.]))
        self.nextrafields = nextrafields

    # Calculate Lorentz invariants from the input four-vectors
    # These four-vectors are either the original jets, or the
    # corresponding combinations
    def forward(self, combvec):
        weighted_e = torch.einsum('ij,bj->bi', self.w_ener,combvec[:, :, 0])
        weighted_pz = torch.einsum('ij,bj->bi', self.w_pid,combvec[:, :, 3])
        weighted_extras = {}
        print("Lola:",self.nextrafields)
        for i in range(self.nextrafields):
            weighted_extras[i] = torch.einsum('ij,bj->bi', self.w_pid,combvec[:, :, 4+i])
        a = combvec[..., :4].unsqueeze(2).repeat(1, 1, self.outputobj, 1)
        b = combvec[..., :4].unsqueeze(1).repeat(1, self.outputobj, 1, 1)
        diff = (a - b)

        distances = torch.einsum('bnmi,ij,bnmj->bnm', diff, self.metric, diff)
        print("Lola:",self.nextrafields)

        weighted_d = torch.einsum('nm,bnm->bn', self.w_dist, distances)
        print("Lola:",self.nextrafields)

        masses = torch.einsum('bni,ij,bnj->bn', combvec[..., :4], self.metric,
                              combvec[..., :4])
        print("Lola:",self.nextrafields)

        ptsq = combvec[:, :, 1]**2 + combvec[:, :, 2]**2
        print("Lola:",self.nextrafields)

        outputs = torch.stack([
            masses,
            ptsq,
            weighted_e,
            weighted_d,
            weighted_pz,
            *weighted_extras.values()
        ], dim=-1)
        print("Lola:",self.nextrafields)

        return outputs


class CoLaLoLa(torch.nn.Module):

    def __init__(self, nobjects, ncombos, noutputs, nscalars=0, nextrafields=0, fflayers=[200]):
        super(CoLaLoLa, self).__init__()
        assert nobjects > 0, "No sense in using CoLaLoLa with only flat variables"
        self.nobjects = nobjects
        self.ntotal = nobjects + ncombos
        self.cola = CoLa(nobjects, ncombos)
        self.lola = LoLa(self.ntotal, nextrafields)
        self.norm = torch.nn.BatchNorm1d(self.ntotal * (5+nextrafields)+nscalars)
        self.head = FeedForwardHead([self.ntotal * (5+nextrafields) + nscalars] + fflayers + [noutputs])
        self.nscalars = nscalars
        self.nextrafields = nextrafields

    def forward(self, vectors):
        #truncate vectors in 4-vectors + scalars, feed jets and merge with scalars after LoLa
        print("In CoLaLoLa")
        print(vectors.shape)
        if self.nscalars:
            scalars = vectors[:,:self.nscalars]
            vectors = vectors[:,self.nscalars:]
            print(scalars.shape)
            print(vectors.shape)
        vectors = vectors.reshape(vectors.shape[0],self.nobjects,4+self.nextrafields)
        print(vectors.shape)
        output = self.cola(vectors)
        print(output.shape)
        output = self.lola(output)
        print(output.shape)
        output = output.reshape(output.shape[0], -1)
        print(output.shape)
        if self.nscalars:
            output = torch.cat([scalars,output],1)
        print(output.shape)
        output = self.norm(output)
        print(output.shape)
        output = self.head(output)
        print(output.shape)

        return output
