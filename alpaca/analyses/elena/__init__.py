import logging
from pathlib import Path
import pandas as pd
import numpy as np

from alpaca.core import BaseMain
from alpaca.batch import BatchManager


log = logging.getLogger(__name__)

def register_cli(subparser, parentparser):

    analysis_name = 'elena'
    analysis_defaults = {
        "Main"       : Elena, #no quotes, pointer to the class
        "outputs"    : 1,
        "categories" : 1,
        "jets"       : 6,
        #"input_files":["truthmatched.h5","truthmatched_unfiltered.h5"],
        #"input_categories":[1,0],
    }

    # Create your own sub-command and add arguments
    parser = subparser.add_parser(analysis_name, parents=[parentparser],
                                   help='Elena\'s analysis sub-command.')
    parser.add_argument('--example', action='store_true',
                        help='example argument')

    return analysis_name, analysis_defaults


class Elena(BaseMain):

    def __init__(self, args):
        super().__init__(args)
        self.bm = BatchManagerElena(
                input_paths=args.input_files,
                input_categories=args.input_categories,
                jets_per_event = args.jets,
                shuffle_events = args.shuffle_events,
            )


class BatchManagerElena(BatchManager):

    @staticmethod
    def get_objects(df, args, **kwargs):
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

        # Get the number of jets per event in the input file by inspecting the
        # second level index of one of the columns
        tot_jets_per_event = len(df['partonindex'].columns.get_level_values('subentry'))
        jets_per_event = args.jets
        zero_jets = args.zero_jets

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
        leadingNarenonzero = df[[("jet_e", i) for i in range(jets_per_event - zero_jets)]].all(axis=1)
        df = df[leadingNincludealltop & leadingNarenonzero]

        maxjets = 10
        jet_vars = ['jet_px','jet_py','jet_pz','jet_e']
        jet_df = df[jet_vars]
        event_stack = np.expand_dims(df["category"].to_numpy(), axis=-1)
        # The input rows have all jet px, all jet py, ... all jet partonindex
        # So segment and swap axes to group by jet
        jet_stack = np.swapaxes(jet_df.values.reshape(len(jet_df), len(jet_vars), maxjets), 1, 2)
        jet_stack = jet_stack[:, :jets_per_event, :]

        return jet_stack, None, None, event_stack, None
