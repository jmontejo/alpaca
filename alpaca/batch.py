import logging

import pandas as pd
import numpy as np
import torch

__all__ = ['BatchManager']


log = logging.getLogger(__name__)


class BatchManager:

    def __init__(self, input_paths, input_categories):
        """Refer to the documentation of the private method `_get_jets`."""
        self._data = {}
        for p,c in zip(input_paths, input_categories):
            self._data[c] = self._get_lorentz_vector(p)

    @staticmethod
    def _get_lorentz_vector(input_path):
        df = pd.read_hdf(input_path, "df")

        # TODO
        # Shoul read the input file and return something like this:
        # lor_vec_stack = np.stack(
        #    [lor_vec.t.regular(),
        #     lor_vec.x.regular(),
        #     lor_vec.y.regular(),
        #     lor_vec.z.regular()],
        #    axis=-1
        # )
        # return lor_vec_stack

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

        lv = []
        cat = []
        for c,v in self._data.items():
            lv.append(v[start_index:stop_index])
            cat.append([c] * N)

        lv = np.concatenate(lv)
        cat = np.concatenate(cat)
        X = torch.as_tensor(lv, dtype=torch.float)
        Y = torch.as_tensor(cat, dtype=torch.float)
        return X, Y

    def get_nr_events(self):
        """Return the length of the internal array which holds all the events.
        """
        # TODO implement it better, this might error if empty data
        #return len(self._data.values()[0])
        return 10**10
