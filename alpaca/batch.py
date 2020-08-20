import logging

import pandas as pd
import numpy as np
import torch

__all__ = ['BatchManager']


log = logging.getLogger(__name__)


class BatchManager:

    n_nonisr = 6
    n_topmatch = 5
    n_btag = 2

    def __init__(self, input_path, shuffle_jets=False, shuffle_events=False,
                 jets_per_event=10, zero_jets=0):
        """Refer to the documentation of the private method `_get_jets`."""
        labeledjets = self._get_jets(
            input_path=input_path,
            shuffle_jets=shuffle_jets,
            shuffle_events=shuffle_events,
            jets_per_event=jets_per_event,
            zero_jets=zero_jets
        )

        # The rest of this method is about parsing the partonindex labels to
        # to derive the truth labels that can be used by the NN.
        # At the end there is also additional sanitisation of the input files
        # which removes some events.

        jets = labeledjets[:, :, :4] #drop parton index, keep 4-vector
        labels = np.array(labeledjets[:, :, -1:].squeeze(), dtype=int) #only parton index

        # Convert the parton labels to bools that the network can make sense of
        # is the jet from the ttbar system?
        jetfromttbar = labels > 0
        # is the jet associated with the top quark?
        # Disregard the jets that are from ISR
        # Account for charge ambiguity by identifying whether the
        # jets match the leading jet or not
        topmatch = np.array([r > 3 for r in labels])
        dummymatch = np.array([r > 4 for r in labels])
        isbjet = np.array([np.equal(r, 1) | np.equal(r, 4) for r in labels])
        jetlabels = np.concatenate([jetfromttbar, topmatch, dummymatch, isbjet], 1)
        # Substitute this line for the preceding if only doing the 6 top jets
        # Not currently configurable by command line because it's a bit more
        # complicated overall + less often changed
        # topmatch = np.array( [ r>3 if r[0]>3 else r<3 for r in labels] )
        # isbjet = np.array( [ np.equal(r,1) | np.equal(r,4) for r in labels] )
        # jetlabels = np.concatenate([jetfromttbar.squeeze(),topmatch[:,1:],isbjet],1)

        self._jets = jets
        self._jetlabels = jetlabels

    @staticmethod
    def _get_jets(input_path, shuffle_jets=False, shuffle_events=False,
                  jets_per_event=10, zero_jets=0):
        """Function that reads an input file and returns a numpy array properly
        formatted, ready to be converted to pytorch tensors.

        Args:
            file_path: path to the input file

            shuffle_jets: if set to `True` for each event the jets are
                shuffled, so that if they were pT ordered they will not be
                anymore

            shuffle_events: if set to `True` shuffles the events order

            jets_per_event: how many jets per event to have in the returned
                object. This corresponds to the size of the inner array. -> can be more than jets in the event, crop last ones if less

            zero_jets: how many of the `jets_per_event` jets to set to zero. If
                you don't shuffle the jets (i.e. `shuffle_jets` set to False)
                the jets will be zero-padded at the end. -> jets_per_event=10 and zero_jets=5, will accept all events with >=5 jets and padd the missing ones

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
        leadingNincludealltop = - df[[("partonindex", i) for i in range(jets_per_event - zero_jets, tot_jets_per_event)]].any(axis=1)
        leadingNarenonzero = df[[("jet_e", i) for i in range(jets_per_event - zero_jets)]].all(axis=1) #complicated cut on at least N jets
        df = df[leadingNincludealltop & leadingNarenonzero]

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

class BatchManager2HDM (BatchManager): #need to include lepton 4-vectors, MET

    def __init__(self, input_path, shuffle_jets=False, shuffle_events=False,
                 jets_per_event=10, zero_jets=0):

        if "truthmatched.h5" in str(input_path):
            return super().__init__(input_path, shuffle_jets, shuffle_events, jets_per_event, zero_jets)

        """Refer to the documentation of the private method `_get_jets`."""
        labeledjets, rest = self._get_jets_and_leps(
            input_path=input_path,
            shuffle_jets=shuffle_jets,
            shuffle_events=shuffle_events,
            jets_per_event=jets_per_event,
            zero_jets=zero_jets
        )

        # The rest of this method is about parsing the partonindex labels to
        # to derive the truth labels that can be used by the NN.
        # At the end there is also additional sanitisation of the input files
        # which removes some events.
        jetsize = 5
        jets = labeledjets[:, :, :jetsize] #drop parton index, keep 4-vector + bjet
        print("BatchManager2HDM",jets_per_event,zero_jets,jets[:5,:,0])

        labels = np.array(labeledjets[:, :, -1:].squeeze(), dtype=int) #only parton index
        lep0 = rest[:,:4]
        lep0 = np.concatenate([lep0,np.zeros([len(lep0),1])],axis=1)
        lep1 = rest[:,4:8]
        lep1 = np.concatenate([lep1,np.zeros([len(lep1),1])],axis=1)
        met  = rest[:,8:]
        met  = np.concatenate([met,np.zeros([len(met),3])],axis=1)

        def myjetlabels(labels):
            myisr = [j==0 for j in labels]
            myfromlep0 = [j==1 for j in labels]
            myfromlep1 = [j==2 for j in labels]
            myfromhad0 = [j==3 for j in labels]
            return np.concatenate([myisr,myfromlep0,myfromlep1,myfromhad0],axis=1) #FIXME

        lep_met = np.stack([lep0,lep1,met],axis=1)
        lep_met_jets = np.concatenate([lep_met,jets],axis=1)

        self._jets = lep_met_jets
        self._jetlabels = myjetlabels(labels) #len njets*4


    @staticmethod
    def _get_jets_and_leps(input_path, shuffle_jets=False, shuffle_events=False,
                           jets_per_event=10, zero_jets=0):
        """See _get_jets"""

        df = pd.read_hdf(input_path, "df")
        # Get the number of jets per event in the input file by inspecting the
        # second level index of one of the columns

        tot_jets_per_event = len(df['jet_e'].columns.get_level_values('subentry'))

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
        leadingNarenonzero = df[[("jet_e", i) for i in range(jets_per_event - zero_jets)]].all(axis=1) #complicated cut on at least N jets
        df = df[leadingNarenonzero]

        if shuffle_events:
            df.reindex(np.random.permutation(df.index))

        # The input rows have all jet px, all jet py, ... all jet partonindex
        # So segment and swap axes to group by jet
        jet_vars = ['jet_pt','jet_eta','jet_phi','jet_e','jet_dl1r','jet_partonindex']
        rest_vars = ['lep0_pt','lep0_eta', 'lep0_phi','lep0_e','lep1_pt','lep1_eta', 'lep1_phi','lep1_e','met','met_phi']
        #jet_vars = ['jet_px','jet_py','jet_pz','jet_e','jet_dl1r','jet_partonindex']
        #rest_vars = ['lep0_px','lep0_py', 'lep0_pz','lep0_e','lep1_px','lep1_py', 'lep1_pz','lep1_e','met_px','met_py','met_pz','met_e']
        maxjets = 8
        #assert len(jet_vars)*2==len(rest_vars) #I'm abusing two leading slots for lep+met
        jet_df = df[jet_vars]
        rest_df = df[rest_vars].droplevel(1,axis=1)
        rest_df = rest_df.loc[:,~rest_df.columns.duplicated()]

        print(jet_df.head())
        print(rest_df.head())

        jet_stack = np.swapaxes(jet_df.values.reshape(len(df), len(jet_vars), maxjets), 1, 2)
        jet_stack = jet_stack[:, :jets_per_event, :]
        #rest_stack = np.swapaxes(rest_df.values.reshape(len(df), len(jet_vars), 2), 1, 2)
        rest_stack = rest_df.values

        if shuffle_jets:
            # shuffle only does the outermost level
            # iterate through rows to shuffle each event individually
            for row in jet_stack:
                np.random.shuffle(row)

        return jet_stack, rest_stack
