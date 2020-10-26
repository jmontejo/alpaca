import logging

import torch
import numpy as np
import pandas as pd

__all__ = ['BatchManager']


log = logging.getLogger(__name__)


class BatchManager:

    def __init__(self, input_paths, dfname="df", input_categories=None, shuffle_jets=False, shuffle_events=False,
                 jets_per_event=10, zero_jets=0, **kwargs):
        """ It is *not* recommended to subclass the constructor, please override only get_objects """

        for input_path, category in zip(input_paths, input_categories):
            df = pd.read_hdf(input_path, dfname)
            df["category" ] = category

            jets, extras, scalars, labels = self.get_objects(
                df,
                jets_per_event=jets_per_event,
                zero_jets=zero_jets,
                **kwargs
            )

        if shuffle_jets:
            # shuffle only does the outermost level
            # iterate through rows to shuffle each event individually
            for row in jets:
                np.random.shuffle(row)

        if shuffle_events:
            p = np.random.permutation(len(labels))
            labels  = labels[p]
            jets    = jets[p]
            extras  = extras[p]
            scalars = scalars[p]

        self._jets    = jets
        self._extras  = extras
        self._scalars = scalars
        self._labels  = labels

        self.build_flat_arrays()

    @staticmethod
    def get_objects(input_paths, input_categories=None, shuffle_jets=False, shuffle_events=False,
                  jets_per_event=10, zero_jets=0):
        ''' Returns a tuple of np.ndarray with the format:
            - jets, extras, scalars, labels
            - jets.shape: (nevents, njets, jetcomponents), jetcomponents is the 4-vector plus possible extra information such as b-tagging score
            - extras.shape: (nevents, nextras, extrascomponents), same as jets but for other objects such as leptons.
                                                                  The extra fields for jets have to appear also in the extras, possibly 0-padded
            - scalars.shape: (nevents, nscalars). Set of event variables such as n_jet, n_bjet, HT, ...
            - labels.shape: (nevents, labelarray) array of target labels. Even if the label is 1-dimensional it still needs to have a shape of (nevents, 1)
            - Some of jets/extras/scalars can be None but at least one has to be filled
        '''
        raise NotImplementedError('Please implement get_objects in your BatchManager','See get_objects in batch.py for documentation')

    def set_valid_events(self,n):
        self.valid_events = n

    def set_valid_fraction(self,f):
        self.valid_events = f*self.get_nr_events()

    def get_valid_batch(self):
        return self.get_torch_batch(self.valid_events)

    def get_train_batch(self, i, total):
        batchevents = self.get_nr_events()//total
        return self.get_torch_batch(batchevents,batchevents*i+self.valid_events)

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

        X = torch.as_tensor(self._flatarrays, dtype=torch.float)
        Y = torch.as_tensor(self._labels, dtype=torch.float)
        return X, Y

    def build_flat_arrays(self):

        arrays = []
        if self._jets is not None: arrays.append(self._jets)
        if self._extras is not None: arrays.append(self._extras)
        if self._scalars is not None:  arrays.append(self._scalars)
        print("arrays",arrays)
        self._flatarrays = np.concatenate([x.reshape(x.shape[0],-1) for x in arrays],axis=1)
        print(self._flatarrays)

    def get_nr_events(self):
        """Return the length of the internal array which holds all the events.
        """
        # self._jets and self._jetslabels should have the same length
        return len(self._flatarrays)

    def is_consistent(self, args):
        objs_njets = self._jets.shape[0] if self._jets is not None else 0
        objs_nextras = self._extras.shape[0] if self._extras is not None else 0

        expected_jets = args.jets
        expected_jet_comp = 4+len(args.extra_jet_fields)
        expected_extras = args.extras
        expected_labels = args.totaloutputs

        obs_jets = self._jets.shape[1] if objs_njets else 0
        obs_jet_comp = self._jets.shape[2] if objs_njets else expected_jet_comp
        obs_extras = self._extras.shape[1] if objs_nextras else 0 
        obs_labels = self._labels.shape[1] if objs_njets else 0

        assert expected_jets == obs_jets, \
                "The number of jets in BatchManager (%d) is not consistent with the expected: %d"%(obs_jets, expected_jets)
        assert expected_jet_comp == obs_jet_comp, \
                "The jet components in BatchManager (%d) is not consistent with the expected: %d"%(obs_jet_comp, expected_jet_comp)
        assert expected_extras == obs_extras, \
                "The number of extra objects in BatchManager (%d) is not consistent with the expected: %d"%(obs_extras, expected_extras)
        assert expected_labels == obs_labels, \
                "The number of labels in BatchManager (%d) is not consistent with the expected: %d"%(obs_labels, expected_labels)

        return True
