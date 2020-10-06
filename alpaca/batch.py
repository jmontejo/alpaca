import logging

import torch

__all__ = ['BatchManager']


log = logging.getLogger(__name__)


class BatchManager:

    def __init__(self, input_paths, input_categories=None, shuffle_jets=False, shuffle_events=False,
                 jets_per_event=10, zero_jets=0):
        """Refer to the documentation of the private method `_get_jets`."""
        jets, labels = self.get_objects(
            input_paths=input_paths,
            input_categories=input_categories,
            shuffle_jets=shuffle_jets,
            shuffle_events=shuffle_events,
            jets_per_event=jets_per_event,
            zero_jets=zero_jets,
        )

        self._jets = jets
        self._jetlabels = labels

    @staticmethod
    def get_objects(input_paths, input_categories=None, shuffle_jets=False, shuffle_events=False,
                  jets_per_event=10, zero_jets=0):
        raise ImplementInSubclass


    def get_torch_batch(self, N, start_index=0):
        """Function that returns pytorch tensors with a batch of events,
        specifically events in the range [start_index:start_index + N].

        Args:
            N: the number of events to return
            start_index: starting index of the internal array

        Return:
            (X, Y): X is in the input to the pytorch model and Y is the truth
                    information associated to X.

        """
        stop_index = start_index + N
        if stop_index > self.get_nr_events():
            log.warning('The stop index is greater than the size of the array')
        X = torch.as_tensor(self._jets[start_index:stop_index, :], dtype=torch.float)
        Y = torch.as_tensor(self._jetlabels[start_index:stop_index,:], dtype=torch.float)
        return X, Y

    def get_nr_events(self):
        """Return the length of the internal array which holds all the events.
        """
        # self._jets and self._jetslabels should have the same length
        return len(self._jets)
