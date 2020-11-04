import logging
from pathlib import Path
import pandas as pd
import numpy as np

from alpaca.core import BaseMain
from alpaca.batch import BatchManager


log = logging.getLogger(__name__)

def register_cli(subparser, parentparser):

    analysis_name = 'allhad_ttbar_chiara'
    analysis_defaults = {
        "Main"       : MainTtbarChiara, #no quotes, pointer to the class
        "extras"     : 0,
        "outputs"    : "n,5,6",
        "jets"       : 7,
        "zero_jets"  : 1,
        "categories" : 3,
        "extra_jet_fields" : [],
    }

    # Create your own sub-command and add arguments
    parser = subparser.add_parser(analysis_name, parents=[parentparser],
                                   help='All-hadronic ttbar (Chiara version) sub-command.')
    parser.add_argument('--not-all-partons', action='store_true')

    return analysis_name, analysis_defaults


class MainTtbarChiara(BaseMain):

    def __init__(self, args):
        super().__init__(args)
        self.train_bm = BatchManagerTtbarChiara(args)
        self.test_bm  = []

    def plots(self):
        log.warning("No plots for MainTtbarChiara")


class BatchManagerTtbarChiara(BatchManager):

    @staticmethod
    def get_objects(df, args, **kwargs):

        print('df.shape')
        print(df.shape)
        # chiara: dirty! remove category
        columns_keep = [c for c in df.columns if 'alpaca' not in c[0]]
        df=df[columns_keep]
        #print(df.columns)
        
        jets_per_event = args.jets
        zero_jets = args.zero_jets
        # chiara: find how to pass these
        all_partons_included=True
        qcd_like=False

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
        #leadingNincludealltop = - df[[("partonindex", i) for i in range(jets_per_event, tot_jets_per_event)]].any(axis=1) #not "- zero_jets" this would make the last jet always ISR
        if qcd_like:
            leadingNincludealltop = (df[[("partonindex", i) for i in range(jets_per_event)]]>0).sum(1) <=2
        else:
            leadingNincludealltop = (df[[("partonindex", i) for i in range(jets_per_event)]]>0).sum(1) ==6

        leadingNarenonzero = df[[("jet_e", i) for i in range(jets_per_event - zero_jets)]].all(axis=1)

        #exactlyleadingNarenonzero = - df[[("jet_e", i) for i in range(jets_per_event, tot_jets_per_event)]].any(axis=1) #not "- zero_jets" this would make the last jet always ISR
        #if all_partons_included:
        #    df = df[leadingNincludealltop & leadingNarenonzero & exactlyleadingNarenonzero]
        #else:
        #    df = df[leadingNarenonzero & exactlyleadingNarenonzero]

        if all_partons_included:
            df = df[leadingNincludealltop & leadingNarenonzero]
        else:
            df = df[leadingNarenonzero]


        # The input rows have all jet px, all jet py, ... all jet partonindex
        # So segment and swap axes to group by jet
        print(df.shape)
        jet_stack = np.swapaxes(df.values.reshape(len(df), 5, 10), 1, 2)
        jet_stack = jet_stack[:, :jets_per_event, :]

        #Reverse to intuitive order
        jet_e = np.copy(jet_stack[:, :, 0])
        jet_px = jet_stack[:, :, 1]
        jet_py = jet_stack[:, :, 2]
        jet_pz = jet_stack[:, :, 3]
        jet_stack[:, :, 0] = jet_px
        jet_stack[:, :, 1] = jet_py
        jet_stack[:, :, 2] = jet_pz
        jet_stack[:, :, 3] = jet_e

        #jet_stack_pt = np.sqrt(jet_stack[:, :, 0]**2+jet_stack[:, :, 1]**2)
        #print(jet_stack_pt.shape)
        #print(np.ones(jet_stack[:, :,0].shape).shape)
        #print(np.maximum(jet_stack_pt,np.ones(jet_stack[:, :,0].shape)))
        #jet_stack_eta = np.arcsinh(jet_stack[:, :, 2]/np.maximum(jet_stack_pt,np.ones(jet_stack[:, :,0].shape)))
        #jet_stack_phi = np.arctan(jet_stack[:, :, 1]/np.maximum(jet_stack[:, :, 0],np.ones(jet_stack[:, :,0].shape)))
        #jet_stack[:, :, 0] = jet_stack_pt
        #jet_stack[:, :, 1] = jet_stack_eta
        #jet_stack[:, :, 2] = jet_stack_phi

        labeledjets = jet_stack

        # The rest of this method is about parsing the partonindex labels to
        # to derive the truth labels that can be used by the NN.
        # At the end there is also additional sanitisation of the input files
        # which removes some events.

        jets = labeledjets[:, :, :4]
        labels = np.array(labeledjets[:, :, -1:].squeeze(), dtype=int)

        #if all_partons_included==False:
        #    self._jets = jets
        #    self._jetlabels = np.zeros((len(jets),jets_per_event+11))
        #    return

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
            if qcd_like: return True

            njets = labeledjets.shape[1]
            return (r[:njets].sum() == 6) and \
                   (r[njets:njets+5].sum() == 2) and \
                   (r[njets+5:].sum() == 2)
        jets_clean = np.array([r for r, t in zip(jets, jetlabels)
                               if good_labels(t,all_partons_included)])
        jetlabels_clean = np.array([r for r in jetlabels if good_labels(r,all_partons_included)])

        return jets_clean,None,None, jetlabels_clean

 

