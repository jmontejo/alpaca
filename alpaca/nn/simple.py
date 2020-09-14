import torch

from alpaca.nn.feedforward import FeedForwardHead

__all__ = ['SimpleNN']


class SimpleNN(torch.nn.Module):

    def __init__(self, nobjects, noutputs, fflayers=[200]):
        super(SimpleNN, self).__init__()
        self.ntotal = nobjects
        self.norm = torch.nn.BatchNorm1d(self.ntotal * 4)
        self.head = FeedForwardHead([self.ntotal * 4] + fflayers + [noutputs])
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total number of trainable parameters in SimpleNN:",pytorch_total_params)

    # Jets go in, labels come out
    def forward(self, vectors):
        output = vectors.reshape(vectors.shape[0], -1)
        output = self.norm(output)
        output = self.head(output)
        return output
