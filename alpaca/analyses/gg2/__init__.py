import logging
from pathlib import Path
import pandas as pd
import numpy as np

from alpaca.core import BaseMain
from alpaca.batch import BatchManager

n_jet_in_input=8 # chiara: can compute this from column names

log = logging.getLogger(__name__)

def register_cli(subparser, parentparser):

    analysis_name = 'gg2'
    analysis_defaults = {
        "Main"       : MainGluGlu, #no quotes, pointer to the class
        "extras"     : 0,
        "outputs"    : "n,5,6",
        "jets"       : 7,
        "zero_jets"  : 1,
        "categories" : 3,
        "extra_jet_fields" : [],
    }

    # Create your own sub-command and add arguments
    parser = subparser.add_parser(analysis_name, parents=[parentparser],
                                   help='Glu Glu (Chiara version) sub-command.')
    parser.add_argument('--not-all-partons', action='store_true')
    parser.add_argument('--no-truth', action='store_true')
    parser.add_argument('--qcd-lambda', type=float, default=0.0, help='Weight factor for qcd mass suppression')
    parser.add_argument('--jet-pT-threshold', type=float, default=0.0, help='Add a threshold to jet selection.')
    parser.add_argument('--first-jet-gluino', action='store_true', help='Assume th efirst jet is a gluino jet. If true, set --outputs "N-1,5,6"')
    parser.add_argument('--multi-class', type=int, default=1, help='Use multiclass classification. If there are 3 classes, set --outputs "N,N,N"')
    parser.add_argument('--flags-for-FSR', action='store_true', help='Gluino flags go up to 12, no longer fixed amount')

    # set as mutually exclusive to plan for other possibilities (e.g. divide by HT)
    scalechoice = parser.add_mutually_exclusive_group()
    scalechoice.add_argument('--scale-e', action='store_true', help='Divide the jet 4-momentum by the sum of the energy of the jets in the event (conside only the jets used in alpaca)')


    return analysis_name, analysis_defaults


class MainGluGlu(BaseMain):

    def __init__(self, args):
        super().__init__(args)
        self.bm    = BatchManagerGluGlu(args)
        if args.input_files_qcd:
            args.input_files = args.input_files_qcd
            self.bmqcd = BatchManagerGluGlu(args, overwritelabels=True)

    def plots(self):
        log.warning("No plots for MainGluGlu")

    def write_output_mass(self, g1m, g2m, normweight,name):
        data = np.stack([g1m,g2m,normweight],axis=1)
        df = pd.DataFrame(data = data, columns=['g1m','g2m','normweight'])
        output_dir = self.get_output_dir() 
        output_name = output_dir / ('massoutput'+name+'.csv')
        df.to_csv(output_name)

    def write_output(self, torch_batch, _P):
        X,Y,spec = torch_batch[0], torch_batch[1], torch_batch[2]

        #_P = P.data.numpy()
        _Y = Y.data.numpy()
        _X = X.data.numpy()
        jet_vars = ['jet_px','jet_py','jet_pz','jet_e']
        extra_fields = set(self.args.extra_jet_fields)
        for extra in extra_fields:
            jet_vars.append(extra)
        col_X = [j+'_'+str(i) for i in range(self.args.jets) for j in jet_vars]
        df_X = pd.DataFrame(data = _X, columns=col_X)
        if spec is not None:
            for i,s in enumerate(self.args.spectators):
                df_X[s]=spec[:,i]
        if Y.shape[1] > 1:
            if self.args.first_jet_gluino:
                if self.args.multi_class > 1:
                    col_P = ['is_ISR_'+str(j) for j in range(self.args.jets-1)]+['is_lead_'+str(j) for j in range(self.args.jets-1)]+['is_sublead_'+str(j) for j in range(self.args.jets-1)]
                else:
                    col_P = ['from_top_'+str(j) for j in range(self.args.jets-1)]+['same_as_lead_'+str(j) for j in range(5)]+['is_b_'+str(j) for j in range(6)]
            elif self.args.multi_class > 1:
                col_P = ['is_ISR_'+str(j) for j in range(self.args.jets)]+['is_lead_'+str(j) for j in range(self.args.jets)]+['is_sublead_'+str(j) for j in range(self.args.jets)]
            else:
                col_P = ['from_top_'+str(j) for j in range(self.args.jets)]+['same_as_lead_'+str(j) for j in range(5)]+['is_b_'+str(j) for j in range(6)]
        else:
            col_P = ['tagged']
        df_P = pd.DataFrame(data = _P, columns=col_P)

        if self.args.no_truth:
            df_test = pd.concat([df_X, df_P], axis=1, sort=False)
        else:
            if self.args.multi_class > 1:
                if self.args.first_jet_gluino:
                    col_Y = ['category_'+str(j) for j in range(self.args.jets-1)]
                else:   
                    col_Y = ['category_'+str(j) for j in range(self.args.jets)]
            else:
                col_Y = [p+'_true' for p in col_P]
            df_Y = pd.DataFrame(data = _Y, columns=col_Y)    
            df_test = pd.concat([df_X, df_P, df_Y], axis=1, sort=False)
        
        output_dir = self.get_output_dir() 
        output_name = output_dir / ('NNoutput_'+self.args.label_roc+'.csv')
        df_test.to_csv(output_name)


class BatchManagerGluGlu(BatchManager):

    @staticmethod
    def get_objects(df, args, **kwargs):

        # remove category -- note: look for better way of doing this
        columns_keep = [c for c in df.columns if 'alpaca' not in c[0]]
        if not args.input_categories:
            df=df[columns_keep]

        print('input df shape', df.shape)

        jets_per_event = args.jets
        zero_jets = args.zero_jets
        use_truth = not args.no_truth
        # when training I need to have and use truth label 
        if args.train: use_truth=True
        if kwargs.get("overwritelabels",False):
            use_truth = False


        # FIXME: pass these as arguments
        # all_partons_included = False
        all_partons_included = not args.not_all_partons
        qcd_like=False

        tot_jets_per_event = len(df['jet_e'].columns.get_level_values('subentry'))

        if (jets_per_event - zero_jets) > tot_jets_per_event:
            log.warning(
                'You are asking for %s jets, but only %s are available',
                jets_per_event - zero_jets,
                tot_jets_per_event
            )
        
	    # Remove the events with the leading jet being ISR
        if args.first_jet_gluino and args.train:
            df = df[ df['partonindex', 0] != 0 ]

        # Drop jets that are below the pT threshold
        df_pT = np.sqrt(df['jet_px']**2 + df['jet_py']**2)
        n_var = len(df.columns.levels[0])
        bool_df = pd.concat([df_pT>args.jet_pT_threshold] * n_var, keys=df.columns.levels[0], axis=1)
        df = df.where(bool_df,0)

        # Select only those events in which
        # - a top jet is not going to be cut, by checking that among the
        #    remaining jets after the Nth there aren't any (note the minus sign)
        # - the leading jets are all existent
        #leadingNincludealltop = - df[[("partonindex", i) for i in range(jets_per_event, tot_jets_per_event)]].any(axis=1) #not "- zero_jets" this would make the last jet always ISR
        leadingNarenonzero = df[[("jet_e", i) for i in range(jets_per_event - zero_jets)]].all(axis=1)
        df = df[leadingNarenonzero]

        if use_truth and all_partons_included:
            if qcd_like:
                leadingNincludealltop = (df[[("partonindex", i) for i in range(jets_per_event)]]>0).sum(1) <=2
            else:
                leadingNincludealltop = (df[[("partonindex", i) for i in range(jets_per_event)]]>0).sum(1) ==6
            df = df[leadingNincludealltop]

        # The input rows have all jet px, all jet py, ... all jet partonindex
        # So segment and swap axes to group by jet
        spectators = []
        for s in args.spectators:            
            if s in [c[0] for c in df.columns]:
                spectators.append(df[s][0]) 
            else:
                spectators.append(np.zeros(len(df)))
        extra_fields = set(args.extra_jet_fields)
        #print('chiara columns')
        #print(df.columns)
        #print([c for c in df.columns if c[0] not in args.spectators and (not 'alpaca' in c[0] and c[1]<n_jet_in_input)])
        df_jets = df[[c for c in df.columns if c[0] not in args.spectators and ('alpaca' not in c[0] and c[1]<n_jet_in_input)]]
        n_comp = 4 + len(extra_fields) # 4 ccomponents for each jet (e, px, py, pz) plus the extra components        
        n_info_jet = n_comp if not use_truth else n_comp+1 # if reading truth, read also parton label
        print('extra fields: ', extra_fields)
        print('ncomp: ', n_comp)
        print('n_info_jet: ', n_info_jet)
        print('n_jet_in_input: ', n_jet_in_input)
        jet_stack = np.swapaxes(df_jets.values.reshape(len(df_jets), n_info_jet, n_jet_in_input), 1, 2)
        jet_stack = jet_stack[:, :jets_per_event, :]
        #Reverse to intuitive order
        jet_e = np.copy(jet_stack[:, :, 0])
        jet_px = jet_stack[:, :, 1]
        jet_py = jet_stack[:, :, 2]
        jet_pz = jet_stack[:, :, 3]

        if args.scale_e:
            # sum of energy of the considered jets in each event
            # keepdims=True needed to be able to broadcast against the original array
            sum_jet_e = np.sum(jet_e, axis=1, keepdims=True)
            jet_e = jet_e/sum_jet_e
            jet_px = jet_px/sum_jet_e
            jet_py = jet_py/sum_jet_e
            jet_pz = jet_pz/sum_jet_e

        #print('jet_e')
        #print(jet_e)
        #print('sum jet_e')
        #print(np.sum(jet_e, axis=1, keepdims=True))

        jet_stack[:, :, 0] = jet_px
        jet_stack[:, :, 1] = jet_py
        jet_stack[:, :, 2] = jet_pz
        jet_stack[:, :, 3] = jet_e
        for i in range(len(extra_fields)):
            jet_stack[:, :, 4+i] = jet_stack[:, :, 4+i]

        #jet_stack_pt = np.sqrt(jet_stack[:, :, 0]**2+jet_stack[:, :, 1]**2)
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

        jets = labeledjets[:, :, :n_comp]
        # print('jets  shape:',jets.shape)

        # signal vs background classification
        if args.input_categories:
            print('Signal vs background classification')
            #print('ncategories', args.ncategories)
            labels = BatchManager.get_event_labels(df, args.ncategories)
            #print(labels.shape)
            #print(labels)
            # chiara 
            labels = labels.values.reshape(len(labels),1)
            #print(labels.shape)
            #print(labels)
            spectators_formatted = np.vstack(spectators).T if(len(args.spectators)>0) else None
            return jets,None,None, labels, spectators_formatted

        # if we arrive here, we are doing event reconstruction
        if use_truth:
            labels = np.array(labeledjets[:, :, -1:].squeeze(), dtype=int) # partonindex
            #if all_partons_included==False:
            #    self._jets = jets
            #    self._jetlabels = np.zeros((len(jets),jets_per_event+11))
            #    return

            if args.multi_class > 1:
                if args.flags_for_FSR:
                    jetlabels = np.zeros_like(labels)
                    firstgindex = np.expand_dims(np.argmax(labels>0, axis=1),axis=1)
                    firstglabel = np.take_along_axis(labels,firstgindex,axis=1)
                    first_gjet_mask = firstglabel <= 6
                    jet_cat_mask = np.where(first_gjet_mask,np.logical_and(labels>0, labels<=6), labels>6)
                    jetlabels[jet_cat_mask] = 1
                    jetlabels[np.logical_and(np.logical_not(jet_cat_mask),labels!=0)] = 2
                    jetlabels.astype('int') 
                else:
                    jetfromttbar = labels > 0
                    maskedlabels = np.ma.masked_where(jetfromttbar == False, labels)
                    nonisrlabels = maskedlabels.compressed().reshape(-1,6)
                    first_gjet_mask = nonisrlabels[:,0]<4
                    print('jetfromttbar shape: ', jetfromttbar.shape)
                    print('maskedlabels shape: ', maskedlabels.shape)
                    print('nonisrlabels shape: ', nonisrlabels.shape)
                    gjet_mask = np.tile(first_gjet_mask[:,np.newaxis],(1,jets_per_event))
                    jet_cat_mask = np.where(gjet_mask,np.logical_and(labels>0, labels<4), labels>3)
                    jetlabels = np.copy(labels)
                    jetlabels[jet_cat_mask] = 1
                    jetlabels[np.logical_and(np.logical_not(jet_cat_mask),labels!=0)] = 2
                    jetlabels.astype('int')
                    if args.first_jet_gluino:
                        jetlabels = np.delete(jetlabels, 0, 1)

            else:
                # Convert the parton labels to bools that the network can make sense of
                # is the jet from the ttbar system?
                jetfromttbar = labels > 0
                # is the jet associated with the top quark?
                # Disregard the jets that are from ISR
                # Account for charge ambiguity by identifying whether the
                # jets match the leading jet or not
                maskedlabels = np.ma.masked_where(jetfromttbar == False, labels)
                nonisrlabels = [r.compressed() for r in maskedlabels]
                # chiara: not ideal, find a better way! 
                for ia,a in enumerate(list(nonisrlabels)):
                    if not a.shape[0] == 6:
                        nonisrlabels[ia] = np.append(nonisrlabels[ia], [0 for i in range(6-a.shape[0])] )
            
                # Labels:
                nonisrlabels = np.array(nonisrlabels)
                # Drop the first element in nonisrlabels
                if args.first_jet_gluino:
                    jetfromttbar = np.delete(jetfromttbar, 0, 1)

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

                if args.multi_class > 1 : 
                    return ((r==1).sum() == (2 if args.first_jet_gluino else 3)) and \
                        ((r==2).sum() == 3)
                
                njets = labeledjets.shape[1]

                if args.first_jet_gluino:
                    return (r[:njets-1].sum() == 5) and \
                        (r[njets-1:njets+5-1].sum() == 2) and \
                        (r[njets+5-1:].sum() == 2)

                return (r[:njets].sum() == 6) and \
                    (r[njets:njets+5].sum() == 2) and \
                    (r[njets+5:].sum() == 2)

            select_clean = np.array([good_labels(r,all_partons_included) for r in jetlabels])
            # print('clean shape:',jets[select_clean].shape)
            # print('leadingNincludealltop', leadingNincludealltop.shape)
            # print('select_clean',select_clean.shape)

            jets_clean = jets[select_clean]
            jetlabels_clean = jetlabels[select_clean]
            spectators_clean = []
            for s in spectators:
                spectators_clean.append(s[select_clean])            
            spectators_formatted_clean = np.vstack(spectators_clean).T if(len(args.spectators)>0) else None
            return jets_clean,None,None, jetlabels_clean, spectators_formatted_clean

        else:
            jetlabels = np.zeros((jets.shape[0], 2))
            spectators_formatted = np.vstack(spectators).T if(len(args.spectators)>0) else None
            return jets,None,None, jetlabels, spectators_formatted
 