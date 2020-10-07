import logging
from pathlib import Path
import pandas as pd
import numpy as np

from alpaca.core import BaseMain
from alpaca.batch import BatchManager


log = logging.getLogger(__name__)

def register_cli(subparser):
    # Create your own sub-command and add arguments
    parser = subparser.add_parser('2hdm',
                                   help='Hello world sub-command.')
    parser.add_argument('--example', action='store_true',
                        help='example argument')
    parser.add_argument('--input-files', '-i', required=True, type=Path,
                        action='append',
                        help='path to the file with the input events')
    parser.add_argument('--shuffle-events', action='store_true')
    parser.add_argument('--shuffle-jets', action='store_true')
    parser.add_argument('--not-all-partons', action='store_true')

    # Set the function corresponding to your subcommand
    parser.set_defaults(Main=Main2HDM)

    return parser


class Main2HDM(BaseMain):

    def __init__(self, args):
        super().__init__(args)
        self.train_bm = BatchManager2HDM(
                input_paths=args.input_files,
                jets_per_event = args.jets,
                shuffle_events = args.shuffle_events,
                all_partons_included=not args.not_all_partons,
                extra_info = args.extra_jet_fields,
            )
        self.test_bm  = []

    def plots(self):
        log.warning("No plots for Main2HDM")


class BatchManager2HDM(BatchManager):

    def __init__(self, input_paths, shuffle_jets=False, shuffle_events=False,
                 jets_per_event=10, zero_jets=0, all_partons_included=True, extra_info=None):
        """Refer to the documentation of the private method `_get_jets`."""
        labeledjets, rest = self.get_objects(
            input_paths=input_paths,
            shuffle_jets=shuffle_jets,
            shuffle_events=shuffle_events,
            jets_per_event=jets_per_event,
            zero_jets=zero_jets,
            extra_info=extra_info,
        )

        # The rest of this method is about parsing the partonindex labels to
        # to derive the truth labels that can be used by the NN.
        # At the end there is also additional sanitisation of the input files
        # which removes some events.
        jetsize = 4+len(extra_info)
        jets = labeledjets[:, :, :jetsize] #drop parton index, keep 4-vector + bjet
        labels = np.array(labeledjets[:, :, -1:].squeeze(), dtype=int) #only parton index

        lep0 = rest[:,:4]
        lep1 = rest[:,4:8]
        met  = rest[:,8:]
        if jetsize == 4:
            pass
        elif jetsize == 5:
            #Fill the btag info for lep+MET outside the DL1r range
            lep0 = np.concatenate([lep0,np.full([len(lep0),1], -7)],axis=1)
            lep1 = np.concatenate([lep1,np.full([len(lep1),1], -7)],axis=1)
            met  = np.concatenate([met, np.full([len(met) ,1], -7)],axis=1)
        else:
            log.error("Weird jet size (%d), this will crash"%jetsize)
            raise WtfIsThis
  
        def myjetlabels(labels):
            myisr = [j!=0 for j in labels]
            myfromlep0 = [j==1 for j in labels]
            myfromlep1 = [j==2 for j in labels]
            myfromhad0 = [j==3 for j in labels]
            return np.concatenate([myisr,myfromlep0,myfromlep1,myfromhad0],axis=1) #FIXME
  
        lep_met = np.stack([lep0,lep1,met],axis=1)
        lep_met_jets = np.concatenate([lep_met,jets],axis=1)
        labels = myjetlabels(labels)
  
        def good_labels(r,all_partons_included):
            if not all_partons_included: return True
  
            njets = labeledjets.shape[1]
            return (r[:njets].sum() == 5) and \
                   (r[njets:njets*2].sum() == 1) and \
                   (r[njets*2:njets*3].sum() == 1) and \
                   (r[njets*3:].sum() == 3)
            #return (r[:njets].sum() >= 3) and \
            #       (r[njets:njets*3].sum() >= 1) and \
            #       (r[njets*3:].sum() >= 2)
  
        lep_met_clean = np.array([r for r,t in zip(lep_met,labels) if good_labels(t,all_partons_included)])
        jets_clean = np.array([r for r,t in zip(jets,labels) if good_labels(t,all_partons_included)])
        labels_clean = np.array([r for r in labels if good_labels(r,all_partons_included)])
  
        self._jets = jets_clean
        self._extras = lep_met_clean
        self._jetlabels = labels_clean #len njets*4
 

    @staticmethod
    def get_objects(input_paths, input_categories=None, shuffle_jets=False, shuffle_events=False,
                  jets_per_event=10, zero_jets=0, extra_info=None):

        df_list = []
        for input_path in input_paths:
            tmp_df = pd.read_hdf(input_path, "df")
            df_list.append(tmp_df)
        df = pd.concat(df_list)
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
  
  
        # The input rows have all jet px, all jet py, ... all jet partonindex
        # So segment and swap axes to group by jet
        #jet_vars = ['jet_pt','jet_eta','jet_phi','jet_e','jet_dl1r','jet_partonindex']
        #rest_vars = ['lep0_pt','lep0_eta', 'lep0_phi','lep0_e','lep1_pt','lep1_eta', 'lep1_phi','lep1_e','met','met_phi']
        jet_vars = ['jet_px','jet_py','jet_pz','jet_e']
        possible_extras = [ 'dl1r']
        unused_info = set(extra_info)
        for extra in possible_extras:
            if extra in extra_info:
                unused_info.remove(extra)
                jet_vars.append('jet_'+extra)
        if len(unused_info):
            log.warning('Some of the extra jet information could not be added %r'%unused_info)

        jet_vars.append('jet_partonindex')
        rest_vars = ['lep0_px','lep0_py', 'lep0_pz','lep0_e','lep1_px','lep1_py', 'lep1_pz','lep1_e','met_px','met_py','met_pz','met_e']
        maxjets = 8
        #assert len(jet_vars)*2==len(rest_vars) #I'm abusing two leading slots for lep+met
        jet_df = df[jet_vars]
        rest_df = df[rest_vars].droplevel(1,axis=1)
        rest_df = rest_df.loc[:,~rest_df.columns.duplicated()]
  
        jet_stack = np.swapaxes(jet_df.values.reshape(len(df), len(jet_vars), maxjets), 1, 2)
        jet_stack = jet_stack[:, :jets_per_event, :]
        #rest_stack = np.swapaxes(rest_df.values.reshape(len(df), len(jet_vars), 2), 1, 2)
        rest_stack = rest_df.values

        return jet_stack, rest_stack

