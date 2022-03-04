import torch

__all__ = ['FeedForwardHead']


class FeedForwardHead(torch.nn.Module):
    def __init__(self, sizes=None, do_multi_class=False):
        super(FeedForwardHead, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(sizes[i-1], sizes[i]) for i in range(1, len(sizes))])
        activations = [torch.nn.ReLU() for i in range(len(sizes) - 2)]
        self.do_multi_class = do_multi_class
        if not self.do_multi_class:
            activations.append(torch.nn.Sigmoid()) 
        self.activations = torch.nn.ModuleList(activations)

    def forward(self, x):
        output = x
        for i in range(len(self.linears)):
            output = self.linears[i](output)
            if self.do_multi_class and i==len(self.linears)-1:
                pass
            else:
                output = self.activations[i](output)
        return output
