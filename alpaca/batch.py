import logging

import pandas as pd
import numpy as np
import torch

__all__ = ['BatchManager']


log = logging.getLogger(__name__)


class BatchManager:

    def __init__(self, input_path, shuffle_jets=False, shuffle_events=False,
                 jets_per_event=10, zero_jets=0, all_partons_included=True):
        """Refer to the documentation of the private method `_get_jets`."""
        labeledjets = self._get_jets(
            input_path=input_path,
            shuffle_jets=shuffle_jets,
            shuffle_events=shuffle_events,
            jets_per_event=jets_per_event,
            zero_jets=zero_jets,
            all_partons_included=all_partons_included,
        )

        # The rest of this method is about parsing the partonindex labels to
        # to derive the truth labels that can be used by the NN.
        # At the end there is also additional sanitisation of the input files
        # which removes some events.

        jets = labeledjets[:, :, :4]
        labels = np.array(labeledjets[:, :, -1:].squeeze(), dtype=int)

        if all_partons_included==False:
            self._jets = jets
            self._jetlabels = np.zeros((len(jets),jets_per_event+11))
            return

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

        # Check that the encoded events have the expected number of positive
        # labels. This cleanup is a consequence of the truth matching script
        # that created the input, so this next bit makes sure that the labels
        # conform to the expectations we have for training, 6 top jets, 2 more
        # to complete top1 and 2 b-jets.
        def good_labels(r,all_partons_included):
            if not all_partons_included: return True
            njets = labeledjets.shape[1]
            return (r[:njets].sum() == 6) and \
                   (r[njets:njets+5].sum() == 2) and \
                   (r[njets+5:].sum() == 2)
        jets_clean = np.array([r for r, t in zip(jets, jetlabels)
                               if good_labels(t,all_partons_included)])
        jetlabels_clean = np.array([r for r in jetlabels if good_labels(r,all_partons_included)])

        self._jets = jets_clean
        self._jetlabels = jetlabels_clean

    @staticmethod
    def _get_jets(input_path, shuffle_jets=False, shuffle_events=False,
                  jets_per_event=10, zero_jets=0, all_partons_included=True):
        """Function that reads an input file and returns a numpy array properly
        formatted, ready to be converted to pytorch tensors.

        Args:
            file_path: path to the input file

            shuffule_jets: if set to `True` for each event the jets are
                shuffled, so that if they were pT ordered they will not be
                anymore

            shuffle_events: if set to `True` shuffles the evets order

            jets_per_event: how many jets per event to have in the returned
                object. This corresponds to the size of the inner array.

            zero_jets: how many of the `jets_per_event` jets to set to zero. If
                you don't shuffle the jets (i.e. `shuffle_jets` set to False)
                the jets will be zero-padded at the end.

        Return:
            jet_stack: numpy array formatted as explained below


        The returned object looks like this (in the case jets_per_event=5, zero_jets=0):

        [[[event 1 jet 0 t, event 1 jet 0 x, event 1 jet 0 y, event 1 jet 0 z, event 1 jet 0 partonindex]
          [event 1 jet 1 t, event 1 jet 1 x, event 1 jet 1 y, event 1 jet 1 z, event 1 jet 1 partonindex]
          ...
          [event 1 jet 5 t, event 1 jet 5 x, event 1 jet 5 y, event 1 jet 5 z, event 1 jet 5 partonindex]]
         [[event 2 jet 1 t, event 2 jet 1 x, event 2 jet 1 y, event 1 jet 1 z, event 2 jet 1 partonindex]
          [event 2 jet 2 t, event 2 jet 2 x, event 2 jet 2 y, event 1 jet 2 z, event 2 jet 2 partonindex]
          ...

        basically they are separated by events. For each event they are
        separated by jet: jet 0, jet 1, etc. And for each jet the five elements
        are the four coordinates of the TLorentz vector: t, x, y, z; plus a
        truth-based parton label that is an integer number betwwen 0 and 6
        (0 = not top jet, 1-3 = b,Wa,Wb from top, 4-6 ditto from antitop).

        """
        df = pd.read_hdf(input_path, "df")
        # Get the number of jets per event in the input file by inspecting the
        # second level index of one of the columns
        tot_jets_per_event = len(df['partonindex'].columns.get_level_values('subentry'))

        if (jets_per_event - zero_jets) > tot_jets_per_event:
            log.warning(
                'You are asking for %s jets, but only %s are available',
                jets_per_event - zero_jets,
                tot_jets_per_event
            )

        # Select only those events in which
        # - a top jet is not going to be cut, by checking that among the
        #    remaining jets after the Nth there aren't any (note the minus sign)
        # - the leading jets are all existent
        leadingNincludealltop = - df[[("partonindex", i) for i in range(jets_per_event, tot_jets_per_event)]].any(axis=1) #not "- zero_jets" this would make the last jet always ISR
        leadingNarenonzero = df[[("jet_e", i) for i in range(jets_per_event - zero_jets)]].all(axis=1)
        if all_partons_included:
            df = df[leadingNincludealltop & leadingNarenonzero]
        else:
            df = df[leadingNarenonzero]

        if shuffle_events:
            df.reindex(np.random.permutation(df.index))

        # The input rows have all jet px, all jet py, ... all jet partonindex
        # So segment and swap axes to group by jet
        jet_stack = np.swapaxes(df.values.reshape(len(df), 5, 10), 1, 2)
        jet_stack = jet_stack[:, :jets_per_event, :]
        if shuffle_jets:
            # shuffle only does the outermost level
            # iterate through rows to shuffle each event individually
            for row in jet_stack:
                np.random.shuffle(row)

        return jet_stack

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
        Y = torch.as_tensor(self._jetlabels[start_index:stop_index, :], dtype=torch.float)
        return X, Y

    def get_nr_events(self):
        """Return the length of the internal array which holds all the events.
        """
        # self._jets and self._jetslabels should have the same length
        return len(self._jets)
