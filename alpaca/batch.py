import logging

import pandas as pd
import numpy as np
import torch

__all__ = ['BatchManager']


log = logging.getLogger(__name__)


class BatchManager:

    def __init__(self, sig_path, shuffle_jets=False, shuffle_events=False,
                 zero_jets=0, jets_per_event=10):
        self._labeledjets = self._get_jets(sig_path, shuffle_jets,
                                          shuffle_events, zero_jets,
                                          jets_per_event)
        log.info('Nr. of signal events: %s', len(self._labeledjets))

    @staticmethod
    def _get_jets(file_path, shuffle_jets=False, shuffle_events=False,
                 zero_jets=0, jets_per_event=10):
        """Function that reads an input file and returns a numpy array properly
        formatted, ready to be converted to pytorch tensors.

        The returned object looks like this:

        [[[event 1 jet 1 t, event 1 jet 1 x, event 1 jet 1 y, event 1 jet 1 z, event 1 jet 1 partonindex]
          [event 1 jet 2 t, event 1 jet 2 x, event 1 jet 2 y, event 1 jet 2 z, event 1 jet 2 partonindex]
            ...
          [event 1 jet 10 t, event 1 jet 10 x, event 1 jet 10 y, event 1 jet 10 z, event 1 jet 10 partonindex]]
          [[event 2 jet 1 t, event 2 jet 1 x, event 2 jet 1 y, event 1 jet 1 z, event 2 jet 1 partonindex]
          [event 2 jet 2 t, event 2 jet 2 x, event 2 jet 2 y, event 1 jet 2 z, event 2 jet 2 partonindex]
           ...

        basically they are separated by events. For each event they are
        separated by jet: jet 1, jet 2, etc. And for each jet the five elements
        are the four coordinates of the TLorentz vector: t, x, y, z; plus a
        truth-based parton label (0=not top jet, 1-3 = b,Wa,Wb from top, 4-6
        ditto from antitop).
        The jets are zero-padded up to the 10th and pT-ordered.
        """

        df = pd.read_hdf(file_path, "df")
        # These next lines can be used to filter out some events, e.g. to
        # limit the training & evaluation to N>6 leading jets
        leadingNcontaintop = df[[("partonindex", i) for i in range(10 - zero_jets, 10)]].sum(axis=1) < 1
        leadingNarenonzero = df[[("jet_e", i) for i in range(10 - zero_jets, 10)]].sum(axis=1) < 1
        df = df[leadingNcontaintop & leadingNarenonzero]
        #
        if shuffle_events:
            df.reindex(np.random.permutation(df.index))
        nevents = len(df)
        # The input rows have all jet px, all jet py, ... all jet partonindex
        # So segment and swap axes to group by jet
        jet_stack = np.swapaxes(df.values.reshape(nevents, 5, 10), 1, 2)
        jet_stack = jet_stack[:, :jets_per_event, :]
        if shuffle_jets:
            # shuffle only does the outermost level
            # iterate through rows to shuffle each event individually
            for row in jet_stack:
                np.random.shuffle(row)
        return jet_stack

    def get_torch_batch(self, N, nlabels, start_index=0):
        """Function that returns pytorch tensors with a batch of events."""
        stop_index = start_index + N
        if stop_index > len(self._labeledjets):
            log.warning('The stop index is greater than the size of the array')
        jets = self._labeledjets[start_index:stop_index, :, :4]
        labels = np.array(self._labeledjets[start_index:stop_index, :nlabels, -1:].squeeze(), dtype=int)

        # Convert the parton labels to bools that the network can make sense of
        # is the jet from the ttbar system?
        jetfromttbar = labels > 0
        # is the jet associated with the top quark?
        # Disregard the jets that are from ISR
        # Account for charge ambiguity by identifying whether the
        # jets match the leading jet or not
        maskedlabels = np.ma.masked_where(jetfromttbar == False, labels)
        nonisrlabels = np.array([r.compressed() for r in maskedlabels])
        topmatch = np.array([r > 3 if r[0] > 3 else r <= 3 for r in nonisrlabels])
        isbjet = np.array([np.equal(r, 1) | np.equal(r, 4) for r in nonisrlabels])
        jetlabels = np.concatenate([jetfromttbar, topmatch[:, 1:], isbjet], 1)
        # Substitute this line for the preceding if only doing the 6 top jets
        # Not currently configurable by command line because it's a bit more
        # complicated overall + less often changed
        # topmatch = np.array( [ r>3 if r[0]>3 else r<3 for r in labels] )
        # isbjet = np.array( [ np.equal(r,1) | np.equal(r,4) for r in labels] )
        # jetlabels = np.concatenate([jetfromttbar.squeeze(),topmatch[:,1:],isbjet],1)

        # Check that the encoded events have the expected number of positive labels
        def good_labels(r):
            return (r[:nlabels].sum() == 6) and \
                   (r[nlabels:nlabels+5].sum() == 2) and \
                   (r[nlabels+5:].sum() == 2)
        jets_clean = np.array([r for r, t in zip(jets, jetlabels)
                               if good_labels(t)])
        jetlabels_clean = np.array([r for r in jetlabels if good_labels(r)])

        X = torch.as_tensor(jets_clean, dtype=torch.float)
        Y = torch.as_tensor(jetlabels_clean, dtype=torch.float)
        return X, Y
