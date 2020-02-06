import torch


class CoLa(torch.nn.Module):
    def __init__(self, nobjects=15, ncombos=20):
        super(CoLa, self).__init__()
        self.nobjects = nobjects
        self.ncombos = ncombos
        self.outputobj = ncombos + nobjects
        self.identity = torch.eye(self.nobjects)
        self.w_combo = torch.nn.Parameter(torch.randn(self.ncombos, self.nobjects))

    def forward(self,vectors):
        combo = torch.cat([self.identity, self.w_combo], dim=0)
        combvec = torch.einsum('ij,bjk->bik', combo, vectors)    
        return combvec


class LoLa(torch.nn.Module):
    def __init__(self, outputobj):
        super(LoLa, self).__init__()
        self.outputobj = outputobj
        self.w_dist = torch.nn.Parameter(torch.randn(self.outputobj, self.outputobj))
        self.w_ener = torch.nn.Parameter(torch.randn(self.outputobj, self.outputobj))
        self.w_pid = torch.nn.Parameter(torch.randn(self.outputobj, self.outputobj))
        self.metric = torch.diag(torch.tensor([1., -1., -1., -1.]))

    def forward(self, combvec):
        weighted_e = torch.einsum('ij,bj->bi', self.w_ener,combvec[:, :, 0])
        weighted_p = torch.einsum('ij,bj->bi', self.w_pid,combvec[:, :, -1])
 
        a = combvec[...,:4].unsqueeze(2).repeat(1, 1, self.outputobj, 1)
        b = combvec[...,:4].unsqueeze(1).repeat(1, self.outputobj, 1, 1)
        diff = (a - b)
        
        distances = torch.einsum('bnmi,ij,bnmj->bnm', diff, self.metric, diff)
        weighted_d = torch.einsum('nm,bnm->bn', self.w_dist, distances)
        masses = torch.einsum('bni,ij,bnj->bn', combvec[..., :4], self.metric, combvec[...,:4])
        ptsq = combvec[:,:,1]**2 + combvec[:,:,2]**2
        outputs = torch.stack([
            masses,
            ptsq,
            weighted_e,
            weighted_d,
            weighted_p,
        ], dim=-1)
        return outputs


class FeedForwardHead(torch.nn.Module):
    def __init__(self, sizes=None):
        super(FeedForwardHead, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(sizes[i-1], sizes[i]) for i in range(1, len(sizes))])
        activations = [torch.nn.ReLU() for i in range(len(sizes) - 2)]
        activations.append(torch.nn.Sigmoid())
        self.activations = torch.nn.ModuleList(activations)

    def forward(self, x):
        output = x
        for i in range(len(self.linears)):
            output = self.linears[i](output)
            output = self.activations[i](output)
        return output


class CoLaLoLa(torch.nn.Module):
    def __init__(self, nobjects, ncombos):
        super(CoLaLoLa, self).__init__()
        self.ntotal = nobjects + ncombos
        self.cola = CoLa(nobjects, ncombos)
        self.lola = LoLa(self.ntotal)
        self.norm = torch.nn.BatchNorm1d(self.ntotal * 5)
        self.head = FeedForwardHead([self.ntotal * 5, 200, 1])
        
    def forward(self,vectors):
        output = self.cola(vectors)
        output = self.lola(output)
        output = output.reshape(output.shape[0], -1)
        output = self.norm(output)
        output = self.head(output)
        return output

