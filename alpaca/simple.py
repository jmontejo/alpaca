import torch


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


class SimpleNN(torch.nn.Module):
    def __init__(self, nobjects, noutputs, fflayers=[200]):
        super(SimpleNN, self).__init__()
        self.ntotal = nobjects
        self.norm = torch.nn.BatchNorm1d(self.ntotal * 4)
        self.head = FeedForwardHead([self.ntotal * 4] + fflayers + [noutputs])

    def forward(self,vectors):
        output = vectors.reshape(vectors.shape[0], -1)
        output = self.norm(output)
        output = self.head(output)
        return output
