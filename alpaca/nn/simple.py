import torch

from alpaca.nn.feedforward import FeedForwardHead

__all__ = ['SimpleNN']


class SimpleNN(torch.nn.Module):

    def __init__(self, nobjects, noutputs, fflayers=[200], do_multi_class=False):
        super(SimpleNN, self).__init__()
        self.ntotal = nobjects
        self.norm = torch.nn.BatchNorm1d(self.ntotal)
        self.head = FeedForwardHead([self.ntotal] + fflayers + [noutputs], do_multi_class)

    # Jets go in, labels come out
    def forward(self, vectors):
        output = vectors.reshape(vectors.shape[0], -1)
        output = self.norm(output)
        output = self.head(output)
        return output
