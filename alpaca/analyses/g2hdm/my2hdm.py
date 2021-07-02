import logging
from pathlib import Path
import pandas as pd
import numpy as np
from copy import deepcopy

from alpaca.core import BaseMain
from alpaca.batch import BatchManager

log = logging.getLogger(__name__)

def register_cli(subparser,parentparser):

    analysis_name = '2hdm'
    analysis_defaults = {
        "Main"       : Main2HDM, #no quotes, pointer to the class
        "extras"     : 3,
        "outputs"    : "n,n,n,n",
        "jets"       : 7,
        "zero_jets"  : 2,
        "categories" : "ISR,lep0,lep1,had0",
        "extra_jet_fields" : ['dl1r'],
        "scalars"    : ["n_jet","n_bjet"],
        "shuffle_events": True,
    }

    # Create your own sub-command and add arguments
    parser = subparser.add_parser(analysis_name, parents=[parentparser],
                                   help='Flavourful 2HDM sub-command.')
    parser.add_argument('--not-all-partons', action='store_true')
    parser.add_argument('--add-reco-mass', action='store_true')
    parser.add_argument('--from-S-labels', action='store_true')
    parser.add_argument('--tq-cat', action='store_true')


    return analysis_name, analysis_defaults


class Main2HDM(BaseMain):

    def __init__(self, args):
        super().__init__(args)
        self.bm = BatchManager2HDM(args)

    def plots(self):
        log.warning("No plots for Main2HDM")

class BatchManager2HDM(BatchManager):

    @staticmethod
    def get_objects(df, args, **kwargs):

        # Get the number of jets per event in the input file by inspecting the
        # second level index of one of the columns
        tot_jets_per_event = len(df['jet_e'].columns.get_level_values('subentry'))
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
        leadingNarenonzero = df[[("jet_e", i) for i in range(jets_per_event - zero_jets)]].all(axis=1) #complicated cut on at least N jets
        df = df[leadingNarenonzero]

  
        # The input rows have all jet px, all jet py, ... all jet partonindex
        # So segment and swap axes to group by jet
        #jet_vars = ['jet_pt','jet_eta','jet_phi','jet_e','jet_dl1r','jet_partonindex']
        #rest_vars = ['lep0_pt','lep0_eta', 'lep0_phi','lep0_e','lep1_pt','lep1_eta', 'lep1_phi','lep1_e','met','met_phi']
        jet_vars = ['jet_px','jet_py','jet_pz','jet_e']
        possible_extras = [ 'dl1r']
        unused_info = set(args.extra_jet_fields)
        for extra in possible_extras:
            if extra in args.extra_jet_fields:
                unused_info.remove(extra)
                jet_vars.append('jet_'+extra)
        if len(unused_info):
            log.warning('Some of the extra jet information could not be added %r'%unused_info)

        jet_vars.append('jet_partonindex')
        jet_vars.append('jet_particleTruthOrigin')
        rest_vars = ['lep0_px','lep0_py', 'lep0_pz','lep0_e','lep1_px','lep1_py', 'lep1_pz','lep1_e','met_px','met_py','met_pz','met_e']
        maxjets = 8
        #assert len(jet_vars)*2==len(rest_vars) #I'm abusing two leading slots for lep+met
        jet_df = df[jet_vars]
        rest_df = df[rest_vars].droplevel(1,axis=1)
        rest_df = rest_df.loc[:,~rest_df.columns.duplicated()]


  
        jet_stack = np.swapaxes(jet_df.values.reshape(len(df), len(jet_vars), maxjets), 1, 2)
        tmpjet_stack = jet_stack[:, :7, :]
        jet_stack = jet_stack[:, :jets_per_event, :]
        #rest_stack = np.swapaxes(rest_df.values.reshape(len(df), len(jet_vars), 2), 1, 2)
        rest_stack = rest_df.values

        # The rest of this method is about parsing the partonindex labels to
        # to derive the truth labels that can be used by the NN.
        # At the end there is also additional sanitisation of the input files
        # which removes some events.
        jetsize = 4+len(args.extra_jet_fields)
        jets = jet_stack[:, :, :jetsize] #drop parton index, keep 4-vector + bjet
        tmpjets = tmpjet_stack[:, :, :jetsize] #drop parton index, keep 4-vector + bjet

        labels = np.array(jet_stack[:, :, -2].squeeze(), dtype=int) #only parton index
        fromSlabels = np.array(jet_stack[:, :, -1].squeeze(), dtype=int) == 15 #only parton index
        mask1 = labels==0
        mask2 = fromSlabels==True
        labels[mask1 & mask2] = 6 #q from tq events

        lep0 = rest_stack[:,:4]
        lep1 = rest_stack[:,4:8]
        met  = rest_stack[:,8:]
        if jetsize == 4:
            pass
        elif jetsize == 5:
            #Fill the btag info for lep+MET outside the DL1r range
            lep0 = np.concatenate([lep0,np.full([len(lep0),1], -7)],axis=1)
            lep1 = np.concatenate([lep1,np.full([len(lep1),1], -7)],axis=1)
            met  = np.concatenate([met, np.full([len(met) ,1], -7)],axis=1)
        else:
            log.error("Weird jet size (%d, %r), this will crash"%(jetsize,args.extra_jet_fields))
            raise WtfIsThis
  
        def myjetlabels(labels):
            myisr = [np.logical_and(j!=0,j!=6) for j in labels]
            myfromlep0 = [j==1 for j in labels]
            myfromlep1 = [j==2 for j in labels]
            myfromhad0 = [j==3 for j in labels]
            return np.concatenate([myisr,myfromlep0,myfromlep1,myfromhad0],axis=1)
        def myjetlabels_tq(labels):
            myisr = [j!=0 for j in labels]
            myfromlep0 = [j==1 for j in labels]
            myfromlep1 = [j==2 for j in labels]
            myfromhad0 = [j==3 for j in labels]
            mytq = [j==6 for j in labels]
            return np.concatenate([myisr,myfromlep0,myfromlep1,myfromhad0,mytq],axis=1)


        lep_met = np.stack([lep0,lep1,met],axis=1)

        if args.from_S_labels:
            labels = fromSlabels
        elif args.tq_cat:
            print(labels)
            print(fromSlabels)
            labels = myjetlabels_tq(labels)
        else:
            labels = myjetlabels(labels)

        def good_labels(r,all_partons_included):
            if not all_partons_included: return True
  
            njets = jet_stack.shape[1]
            return (r[:njets].sum() == 5) and \
                   (r[njets:njets*2].sum() == 1) and \
                   (r[njets*2:njets*3].sum() == 1) and \
                   (r[njets*3:].sum() == 3)

        lep_met_clean = np.array([r for r,t in zip(lep_met,labels) if good_labels(t,not args.not_all_partons)])
        jets_clean = np.array([r for r,t in zip(jets,labels) if good_labels(t,not args.not_all_partons)])
        tmpjets_clean = np.array([r for r,t in zip(tmpjets,labels) if good_labels(t,not args.not_all_partons)])
        labels_clean = np.array([r for r in labels if good_labels(r,not args.not_all_partons)])

        if args.add_reco_mass:

            fakeargs = deepcopy(args)
            fakeargs.add_reco_mass = False
            fakeargs.normalize_scores = False
            fakeargs.normalize_scores_ISR = False
            fakeargs.jets = 7
            fakeargs.extras = 3
            fakeargs.scalars = ["n_jet","n_bjet"]
            fakeargs.input_files = [args.current[0]]
            fakeargs.input_categories = [args.current[1]]
            bm = BatchManager2HDM(fakeargs,**kwargs)
            param_file = '/Users/JMontejo/Documents/Physics/Machine_learning/alpaca/run_matchS/data/alpaca_MSX_bis/NN.pt'
            import torch
            model = torch.load(param_file)
            model.eval()
            test_torch_batch = bm.get_torch_batch(len(labels))
            X,Y = test_torch_batch[0], test_torch_batch[1]
            mass_list=[]

            fakeargs.mass_reco_choice=0
            fakeargs.jet_assign_choice=4
            from torch.utils.data import DataLoader
            for i, batch in enumerate(DataLoader(X, batch_size=500)):
                P = model(batch)
                P = P.data.numpy()
                jets_lep_met = batch[:,:50].reshape(-1,10,5).data.numpy() #drop the scalars

                mask1 = np.abs(jets_lep_met[:, 0, 3] - 1.13049576e+05) < 1
                mask2 = jets_lep_met[:, 8, 3] == 0 
                mask = mask1 & mask2
                if np.any(mask):
                    print("Will evaluate model")
                    print(jets_lep_met[mask])
                    print(batch[mask])
                    print(P[mask])

                mass, perfect = do_reco(fakeargs, jets_lep_met*1e-3, P[:,:fakeargs.jets], P[:,fakeargs.jets:fakeargs.jets*2], P[:,fakeargs.jets*2:fakeargs.jets*3], P[:,fakeargs.jets*3:])
                mass_list.append(mass)
            mass = np.concatenate(mass_list, axis=0)

            print("X",X[:2])
            print(mass[:2])

        if args.scalars:
            print("lep_met_clean",lep_met_clean[:2])
            if args.add_reco_mass:
                print(mass[:2])
                df["reco_mass"] = mass
            scalar_df = df[args.scalars].droplevel(1,axis=1)
            scalar_df = scalar_df.loc[:,~scalar_df.columns.duplicated()]

            scalar_stack = scalar_df.values
            scalars_clean = np.array([r for r,t in zip(scalar_stack,labels) if good_labels(t,not args.not_all_partons)])

        else:
            scalars_clean = None

        if args.extras == 0:
            lep_met_clean = None
        if args.jets == 0:
            jets_clean = None

        return jets_clean, lep_met_clean, scalars_clean, labels_clean, None


def minv(a):
    return np.sqrt(np.maximum(np.zeros(len(a)),a[:,3]**2 - a[:,2]**2 - a[:,1]**2 - a[:,0]**2))
def pt(a):
    return np.sqrt(a[:,1]**2 + a[:,0]**2)

def do_reco(args, 
    jets,
    ISR_pred_score, 
    lep0_pred_score,
    lep1_pred_score,
    had0_pred_score,
    ISR_truth=None, 
    lep0_truth=None,
    lep1_truth=None,
    had0_truth=None):


    ISR_pred_score  = np.ones_like(ISR_pred_score) - ISR_pred_score

    nevents = len(jets)

    if args.normalize_scores:
        sum_score = had0_pred_score+lep0_pred_score+lep1_pred_score
    elif args.normalize_scores_ISR:
        sum_score = had0_pred_score+lep0_pred_score+lep1_pred_score+ISR_pred_score
    else:
        sum_score = 1
    had0_pred_score /= sum_score
    lep0_pred_score /= sum_score
    lep1_pred_score /= sum_score
    ISR_pred_score /= sum_score
    
    #Some events will have zero leptonic tops, need to FIXME
    lep0_choice = (lep0_pred_score.max(axis=1,keepdims=1) == lep0_pred_score) & (lep0_pred_score > ISR_pred_score) & (lep0_pred_score > had0_pred_score) & (lep0_pred_score > lep1_pred_score)
    lep1_choice = (lep1_pred_score.max(axis=1,keepdims=1) == lep1_pred_score) & (lep1_pred_score > ISR_pred_score) & (lep1_pred_score > had0_pred_score) & (lep1_pred_score > lep0_pred_score)
    had0_choice = np.zeros_like(had0_pred_score, dtype=bool)
    
    #print ((lep0_choice | lep1_choice).shape)
    #print(had0_pred_score.shape)
    #print(had0_pred_score)
    had0_pred_score[lep0_choice] = 0
    had0_pred_score[lep1_choice] = 0
    #print(had0_pred_score)

    if args.jet_assign_choice == 0: #category with max value
        had0_choice = (had0_pred_score > ISR_pred_score) & (had0_pred_score > lep0_pred_score) & (had0_pred_score > lep1_pred_score)
    elif args.jet_assign_choice == 1: #max 3 had tops
        _had0_pred_score = np.array(had0_pred_score) # temp var to modify
        usenjet = 3
        for i in range(usenjet):
            # choose the highest scoring jet
            had0_choice[np.arange(nevents), _had0_pred_score.argmax(1)] = True
            # set the score to 0 to ignore it in the next iteration
            _had0_pred_score[np.arange(nevents), _had0_pred_score.argmax(1)] = 0
        had0_choice = had0_choice & (had0_pred_score > ISR_pred_score) & (had0_pred_score > lep0_pred_score) & (had0_pred_score > lep1_pred_score)
    elif args.jet_assign_choice == 2: #at least 1 lep 2 had tops
        njets = lep0_pred_score.shape[1]
        mask = np.repeat(lep0_pred_score.max(axis=1,keepdims=True) > lep1_pred_score.max(axis=1,keepdims=True),njets,axis=1)
        #event = 3
        #print(lep0_pred_score[event])
        #print(lep1_pred_score[event])
        #print(mask[event])
        #print(lep0_choice[event])
        #print(lep1_choice[event])
        lep0_choice = lep0_choice | ((lep0_pred_score.max(axis=1,keepdims=1) == lep0_pred_score)) & mask
        lep1_choice = lep1_choice | ((lep1_pred_score.max(axis=1,keepdims=1) == lep1_pred_score)) & (~mask)
        #mass = np.where(pt(reco0) > pt(reco1), minv(reco0), minv(reco1))
        had0_pred_score[lep0_choice] = 0
        had0_pred_score[lep1_choice] = 0
        _had0_pred_score = np.array(had0_pred_score)
        usenjet = 2
        for i in range(usenjet):
            had0_choice[np.arange(nevents), _had0_pred_score.argmax(1)] = True
            _had0_pred_score[np.arange(nevents), _had0_pred_score.argmax(1)] = 0
        had0_choice = had0_choice | (had0_pred_score > ISR_pred_score) & (had0_pred_score > lep0_pred_score) & (had0_pred_score > lep1_pred_score)
    elif args.jet_assign_choice == 3: #at least 2 lep 2 had tops
        lep0_choice[np.arange(nevents), lep0_pred_score.argmax(1)] = True
        lep1_choice[np.arange(nevents), lep1_pred_score.argmax(1)] = True
        had0_pred_score[lep0_choice] = 0
        had0_pred_score[lep1_choice] = 0
        _had0_pred_score = np.array(had0_pred_score)
        usenjet = 2
        for i in range(usenjet):
            had0_choice[np.arange(nevents), _had0_pred_score.argmax(1)] = True
            _had0_pred_score[np.arange(nevents), _had0_pred_score.argmax(1)] = 0
        had0_choice = had0_choice | (had0_pred_score > ISR_pred_score) & (had0_pred_score > lep0_pred_score) & (had0_pred_score > lep1_pred_score)
    elif args.jet_assign_choice == 4: #exactly 2 lep 3 had tops
        lep0_choice[np.arange(nevents), lep0_pred_score.argmax(1)] = True
        lep1_choice[np.arange(nevents), lep1_pred_score.argmax(1)] = True
        had0_pred_score[lep0_choice] = 0
        had0_pred_score[lep1_choice] = 0
        _had0_pred_score = np.array(had0_pred_score)
        usenjet = 3
        for i in range(usenjet):
            had0_choice[np.arange(nevents), _had0_pred_score.argmax(1)] = True
            _had0_pred_score[np.arange(nevents), _had0_pred_score.argmax(1)] = 0

    choice0 = np.concatenate([np.tile([1,0,0],(len(had0_choice),1)), had0_choice | lep0_choice], axis=1)
    choice1 = np.concatenate([np.tile([0,1,0],(len(had0_choice),1)), had0_choice | lep1_choice], axis=1)

    #print("On average lep0 jets:",lep0_choice.sum(1).mean())
    #print("On average lep1 jets:",lep1_choice.sum(1).mean())
    #print("On average had0 jets:",had0_choice.sum(1).mean())

    if had0_truth is not None:
        perfect = np.logical_xor(had0_truth, had0_choice) | np.logical_xor(lep0_truth, lep0_choice) | np.logical_xor(lep1_truth, lep1_choice)
        perfect = perfect.sum(1)==0
        #print("Perfect fraction",perfect.mean())
    else:
        perfect = None

    jets_choice0 = np.copy(jets)
    jets_choice1 = np.copy(jets)
    jets_choice0[~choice0.astype(np.bool)] = 0
    jets_choice1[~choice1.astype(np.bool)] = 0
    reco0 = jets_choice0.sum(1)
    reco1 = jets_choice1.sum(1)

    if args.mass_reco_choice == 0: #min mass
        mass = np.where(minv(reco0) < minv(reco1), minv(reco0), minv(reco1))
    elif args.mass_reco_choice == 1: #max mass
        mass = np.where(minv(reco0) > minv(reco1), minv(reco0), minv(reco1))
    elif args.mass_reco_choice == 2: #min pt mass
        mass = np.where(pt(reco0) < pt(reco1), minv(reco0), minv(reco1))
    elif args.mass_reco_choice == 3: #max pt mass
        mass = np.where(pt(reco0) > pt(reco1), minv(reco0), minv(reco1))

    mask1 = np.abs(jets[:, 0, 3] - 1.13049576e+02) < 1e-3
    mask2 = jets[:, 8, 3] == 0 
    mask = mask1 & mask2
    if np.any(mask):
        print("Will evaluate do_reco")
        print(jets[mask])
        print(ISR_pred_score[mask])
        print(lep0_pred_score[mask])
        print(lep1_pred_score[mask])
        print(had0_pred_score[mask])
        print(mass[mask])

    return mass, perfect

def do_reco_ttq(args, 
    jets,
    ISR_pred_score, 
    lep0_pred_score,
    lep1_pred_score,
    had0_pred_score,
    tq_pred_score,
    ISR_truth=None, 
    lep0_truth=None,
    lep1_truth=None,
    had0_truth=None,
    tq_truth=None):


    ISR_pred_score  = np.ones_like(ISR_pred_score) - ISR_pred_score

    nevents = len(jets)

    if args.normalize_scores:
        sum_score = had0_pred_score+lep0_pred_score+lep1_pred_score+tq_pred_score
    elif args.normalize_scores_ISR:
        sum_score = had0_pred_score+lep0_pred_score+lep1_pred_score+tq_pred_score+ISR_pred_score
    else:
        sum_score = 1
    had0_pred_score /= sum_score
    lep0_pred_score /= sum_score
    lep1_pred_score /= sum_score
    tq_pred_score  /= sum_score
    ISR_pred_score /= sum_score


    print(ISR_pred_score[:3])
    print(lep0_pred_score[:3])
    print(lep1_pred_score[:3])
    print(had0_pred_score[:3])
    print(tq_pred_score[:3])
    
    #Some events will have zero leptonic tops, need to FIXME
    lep0_choice = (lep0_pred_score.max(axis=1,keepdims=1) == lep0_pred_score) & (lep0_pred_score > ISR_pred_score) & (lep0_pred_score > had0_pred_score) & (lep0_pred_score > lep1_pred_score) & (lep0_pred_score > tq_pred_score)
    lep1_choice = (lep1_pred_score.max(axis=1,keepdims=1) == lep1_pred_score) & (lep1_pred_score > ISR_pred_score) & (lep1_pred_score > had0_pred_score) & (lep1_pred_score > lep0_pred_score) & (lep1_pred_score > tq_pred_score)
    tq_choice   = (tq_pred_score.max(axis=1,keepdims=1)   == tq_pred_score)   & (tq_pred_score   > ISR_pred_score) & (tq_pred_score   > had0_pred_score) & (tq_pred_score   > lep0_pred_score) & (tq_pred_score   > lep1_pred_score)
    #Some events will have zero leptonic tops, need to FIXME
    #lep0_choice = (lep0_pred_score.max(axis=1,keepdims=1) == lep0_pred_score) & (lep0_pred_score > had0_pred_score) & (lep0_pred_score > lep1_pred_score) & (lep0_pred_score > tq_pred_score)
    #lep1_choice = (lep1_pred_score.max(axis=1,keepdims=1) == lep1_pred_score) & (lep1_pred_score > had0_pred_score) & (lep1_pred_score > lep0_pred_score) & (lep1_pred_score > tq_pred_score)
    #tq_choice   = (tq_pred_score.max(axis=1,keepdims=1)   == tq_pred_score)   & (tq_pred_score   > had0_pred_score) & (tq_pred_score   > lep0_pred_score) & (tq_pred_score   > lep1_pred_score)

    #print ((lep0_choice | lep1_choice).shape)
    #print(had0_pred_score.shape)
    #print(had0_pred_score)
    had0_pred_score[lep0_choice] = 0
    had0_pred_score[lep1_choice] = 0
    had0_pred_score[tq_choice] = 0
    #print(had0_pred_score)

    print(lep0_choice[:3])
    print(lep1_choice[:3])
    print(tq_choice[:3])
    tq_choice[tq_choice.sum(1)==0,np.argmax(had0_pred_score>0)] = 1 #take leading non-assigned for those without a choice
    print(tq_choice[:3])

    print(ISR_truth[:3])
    print(lep0_truth[:3])
    print(lep1_truth[:3])
    print(had0_truth[:3])
    print(tq_truth[:3])

    choice0 = np.concatenate([np.tile([1,0,0],(len(tq_choice),1)), tq_choice | lep0_choice], axis=1)
    choice1 = np.concatenate([np.tile([0,1,0],(len(tq_choice),1)), tq_choice | lep1_choice], axis=1)

    #print("On average lep0 jets:",lep0_choice.sum(1).mean())
    #print("On average lep1 jets:",lep1_choice.sum(1).mean())
    #print("On average had0 jets:",had0_choice.sum(1).mean())


    if had0_truth is not None:
        perfect = np.logical_xor(tq_truth, tq_choice) | np.logical_xor(lep0_truth, lep0_choice) | np.logical_xor(lep1_truth, lep1_choice)
        perfect = perfect.sum(1)==0
    else:
        perfect = None

    jets_choice0 = np.copy(jets)
    jets_choice1 = np.copy(jets)
    jets_choice0[~choice0.astype(np.bool)] = 0
    jets_choice1[~choice1.astype(np.bool)] = 0
    reco0 = jets_choice0.sum(1)
    reco1 = jets_choice1.sum(1)

    if args.mass_reco_choice == 0: #min mass
        mass = np.where(minv(reco0) < minv(reco1), minv(reco0), minv(reco1))
    elif args.mass_reco_choice == 1: #max mass
        mass = np.where(minv(reco0) > minv(reco1), minv(reco0), minv(reco1))
    elif args.mass_reco_choice == 2: #min pt mass
        mass = np.where(pt(reco0) < pt(reco1), minv(reco0), minv(reco1))
    elif args.mass_reco_choice == 3: #max pt mass
        mass = np.where(pt(reco0) > pt(reco1), minv(reco0), minv(reco1))

    return mass, perfect



def do_reco_fromS(args, 
    jets,
    fromS_pred_score, 
    fromS_truth=None):

    nevents = len(jets)

    
    #Some events will have zero leptonic tops, need to FIXME
    lep0_choice = (lep0_pred_score.max(axis=1,keepdims=1) == lep0_pred_score) & (lep0_pred_score > ISR_pred_score) & (lep0_pred_score > had0_pred_score) & (lep0_pred_score > lep1_pred_score)
    lep1_choice = (lep1_pred_score.max(axis=1,keepdims=1) == lep1_pred_score) & (lep1_pred_score > ISR_pred_score) & (lep1_pred_score > had0_pred_score) & (lep1_pred_score > lep0_pred_score)
    tq_choice = np.zeros_like(had0_pred_score, dtype=bool)
    
    #print ((lep0_choice | lep1_choice).shape)
    #print(had0_pred_score.shape)
    #print(had0_pred_score)
    had0_pred_score[lep0_choice] = 0
    had0_pred_score[lep1_choice] = 0
    #print(had0_pred_score)

    if args.jet_assign_choice == 0: #leading non-leptonic jet
        tq_choice[np.argmax(had0_pred_score>0)] = 1 #argmax returns the first in case of tie

    choice0 = np.concatenate([np.tile([1,0,0],(len(tq_choice),1)), tq_choice | lep0_choice], axis=1)
    choice1 = np.concatenate([np.tile([0,1,0],(len(tq_choice),1)), tq_choice | lep1_choice], axis=1)

    #print("On average lep0 jets:",lep0_choice.sum(1).mean())
    #print("On average lep1 jets:",lep1_choice.sum(1).mean())
    #print("On average had0 jets:",had0_choice.sum(1).mean())


    perfect = None

    jets_choice0 = np.copy(jets)
    jets_choice1 = np.copy(jets)
    jets_choice0[~choice0.astype(np.bool)] = 0
    jets_choice1[~choice1.astype(np.bool)] = 0
    reco0 = jets_choice0.sum(1)
    reco1 = jets_choice1.sum(1)

    if args.mass_reco_choice == 0: #min mass
        mass = np.where(minv(reco0) < minv(reco1), minv(reco0), minv(reco1))
    elif args.mass_reco_choice == 1: #max mass
        mass = np.where(minv(reco0) > minv(reco1), minv(reco0), minv(reco1))
    elif args.mass_reco_choice == 2: #min pt mass
        mass = np.where(pt(reco0) < pt(reco1), minv(reco0), minv(reco1))
    elif args.mass_reco_choice == 3: #max pt mass
        mass = np.where(pt(reco0) > pt(reco1), minv(reco0), minv(reco1))

    return mass, perfect

