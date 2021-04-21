import logging

import torch
import numpy as np
import pandas as pd

__all__ = ['BatchManager']


log = logging.getLogger(__name__)


class BatchManager:

    internal_category_name = "alpaca_category"

    def __init__(self, args, dfname="df", **kwargs):
        """ It is *not* recommended to subclass the constructor, please override only get_objects """

        from itertools import zip_longest
        tmp_jets = []
        tmp_extras = []
        tmp_scalars = []
        tmp_labels = [] 
        tmp_spectators = []

        for input_path, category in zip_longest(args.input_files, args.input_categories):
            print(input_path)
            print(dfname)            
            df = pd.read_hdf(input_path, dfname)
            df[self.internal_category_name] = category
            df = df.replace(True, 1)
            df = df.replace(False, 0)
            print('shape',df.shape[0])
            #if category == 0:
            #    print('category 0')
            #    keep_index = np.random.random(df.shape[0]) < 0.1
            #    df = df[keep_index]
            print('new shape',df.shape[0])

            jets, extras, scalars, labels, spectators = self.get_objects(
                df, args,
                **kwargs
            )

            if jets is not None: tmp_jets.append(jets)
            if extras is not None: tmp_extras.append(extras)
            if scalars is not None: tmp_scalars.append(scalars)
            tmp_labels.append(labels)
            if spectators is not None: tmp_spectators.append(spectators)

        jets = np.concatenate(tmp_jets) if tmp_jets else None
        extras = np.concatenate(tmp_extras) if tmp_extras else None
        scalars = np.concatenate(tmp_scalars) if tmp_scalars else None
        labels = np.concatenate(tmp_labels)
        spectators = np.concatenate(tmp_spectators) if tmp_spectators else None

        if args.shuffle_jets and jets is not None:
            # shuffle only does the outermost level
            # iterate through rows to shuffle each event individually
            for row in jets:
                np.random.shuffle(row)

        if args.shuffle_events:
            p = np.random.permutation(len(labels))
            labels  = labels[p]
            if jets is not None: jets    = jets[p]
            if extras is not None: extras  = extras[p]
            if scalars is not None: scalars = scalars[p]
            if spectators is not None: spectators = spectators[p]

        self._jets    = jets
        self._extras  = extras
        self._scalars = scalars
        self._labels  = labels
        self._spectators = spectators

        self.build_flat_arrays()

    @staticmethod
    def write_output(torch_batch, P):
        ''' Writes the result into a file 
            Takes as input the return value of get_torch_batch and P = model(X)
        '''
        raise NotImplementedError('Please implement write_output in your BatchManager','See write_output in batch.py for documentation')

    @staticmethod
    def get_objects(df, args, **kwargs):
        ''' Returns a tuple of np.ndarray with the format:
            - jets, extras, scalars, labels, spectators
            - jets.shape: (nevents, njets, jetcomponents), jetcomponents is the 4-vector plus possible extra information such as b-tagging score
            - extras.shape: (nevents, nextras, extrascomponents), same as jets but for other objects such as leptons.
                                                                  The extra fields for jets have to appear also in the extras, possibly 0-padded
            - scalars.shape: (nevents, nscalars). Set of event variables such as n_jet, n_bjet, HT, ...
            - labels.shape: (nevents, labelarray) array of target labels. Even if the label is 1-dimensional it still needs to have a shape of (nevents, 1)
            - spectators.shape: (nevents, nspectators). Spectator variables for each event, assumed to be scalar
            - Some of jets/extras/scalars can be None but at least one has to be filled
        '''
        raise NotImplementedError('Please implement get_objects in your BatchManager','See get_objects in batch.py for documentation')

    @staticmethod
    def get_event_labels(df, ncategories):
        assert ncategories>=1 and type(ncategories)==int
        if ncategories == 1:
            return df[BatchManager.internal_category_name]
        else:
            cats = range(ncategories)
            return np.stack([df[BatchManager.internal_category_name] == c for c in cats],axis=1)

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

        #import sys
        #np.set_printoptions(threshold=sys.maxsize)
        #print(self._flatarrays[0:25,:].dtype)
        #print(self._flatarrays[0:25,:])

        X = torch.as_tensor(self._flatarrays[start_index:stop_index, :], dtype=torch.float)
        Y = torch.as_tensor(self._labels[start_index:stop_index, :], dtype=torch.float)
        if self._spectators is not None:
            return X, Y, self._spectators[start_index:stop_index]
        else:
            return X, Y

    def build_flat_arrays(self):

        arrays = []
        if self._extras is not None: arrays.append(self._extras)
        if self._jets is not None: arrays.append(self._jets)
        if self._scalars is not None:  arrays.append(self._scalars)
        self._flatarrays = np.concatenate([x.reshape(x.shape[0],-1) for x in arrays],axis=1)

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
        obs_labels = self._labels.shape[1]

        assert expected_jets == obs_jets, \
                "The number of jets in BatchManager (%d) is not consistent with the expected: %d"%(obs_jets, expected_jets)
        assert expected_jet_comp == obs_jet_comp, \
                "The jet components in BatchManager (%d) is not consistent with the expected: %d"%(obs_jet_comp, expected_jet_comp)
        assert expected_extras == obs_extras, \
                "The number of extra objects in BatchManager (%d) is not consistent with the expected: %d"%(obs_extras, expected_extras)
        assert expected_labels == obs_labels, \
                "The number of labels in BatchManager (%d) is not consistent with the expected: %d"%(obs_labels, expected_labels)

        test_events = 100
        if objs_njets:
            assert np.all(self._jets[:test_events,:,3]>=0), \
                "Negative entries in the fourth jet component. The expected order is Px/Py/Pz/E"
            assert np.any(self._jets[:test_events,:,0]<0), \
                "No negative entries in the first jet component. The expected order is Px/Py/Pz/E"

        return True
