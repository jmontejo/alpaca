from pathlib import Path
import pandas as pd
import numpy as np

from alpaca.core import BaseMain
from alpaca.batch import BatchManager


def register_cli(subparser):
    # Create your own sub-command and add arguments
    parser = subparser.add_parser('elena',
                                   help='Hello world sub-command.')
    parser.add_argument('--example', action='store_true',
                        help='example argument')
    parser.add_argument('--input-files', '-i', required=True, type=Path,
                        action='append',
                        help='path to the file with the input events')
    parser.add_argument('--input-categories', '-ic', required=True, type=int,
                        action='append',
                        help='path to the file with the input events')
    parser.add_argument('--shuffle-events', action='store_true')
    parser.add_argument('--shuffle-jets', action='store_true')

    # Set the function corresponding to your subcommand
    parser.set_defaults(Main=Elena)

    return parser


class Elena(BaseMain):

    def __init__(self, args):
        super().__init__(args)
        self.train_bm = BatchManagerElena(
                input_paths=args.input_files,
                input_categories=args.input_categories,
                jets_per_event = args.jets,
                shuffle_events = args.shuffle_events,
            )
        self.test_bm  = []


class BatchManagerElena(BatchManager):

    @staticmethod
    def get_objects(input_paths, input_categories=None, shuffle_jets=False, shuffle_events=False,
                  jets_per_event=10, zero_jets=0):
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
        df_list = []
        for input_path, category in zip(input_paths, input_categories):
            tmp_df = pd.read_hdf(input_path, "df")
            tmp_df["category" ] = category
            df_list.append(tmp_df)
            print(input_path, category)
        df = pd.concat(df_list)
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
        if shuffle_jets:
            # shuffle only does the outermost level
            # iterate through rows to shuffle each event individually
            for row in jet_stack:
                np.random.shuffle(row)

        if shuffle_events:
            p = np.random.permutation(len(event_stack))
            event_stack = event_stack[p]
            jet_stack = jet_stack[p]

        return jet_stack, event_stack
