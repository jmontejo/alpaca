import torch

__all__ = ['FeedForwardHead']


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
