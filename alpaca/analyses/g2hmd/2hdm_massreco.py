import logging
from pathlib import Path
import pandas as pd
import numpy as np

from alpaca.core import BaseMain
from alpaca.analyses.g2hmd.my2hmd import BatchManager2HDM

log = logging.getLogger(__name__)

def register_cli(subparser,parentparser):

    analysis_name = '2hdm_massreco'
    analysis_defaults = {
        "Main"       : Eval2HDM, #no quotes, pointer to the class
        "extras"     : 3,
        "outputs"    : "n,n,n,n",
        "jets"       : 7,
        "zero_jets"  : 1,
        "categories" : 4,
        "extra_jet_fields" : ['dl1r'],
        "scalars"    : ["n_jet","n_bjet"],
        "shuffle_events": True,
    }

    # Create your own sub-command and add arguments
    parser = subparser.add_parser(analysis_name, parents=[parentparser],
                                   help='Flavourful 2HDM sub-command.')
    parser.add_argument('--npz', type=Path,
                        help='path to the npz file out of alpaca')
    parser.add_argument('--small', action="store_true")
    parser.add_argument('--allow-2jet', action="store_true",
                    help='Use two or more jets for the tops')
    parser.add_argument('--normalize-scores', action="store_true",
                    help='Renormalize to sum of scores')
    parser.add_argument('--normalize-scores-ISR', action="store_true",
                    help='Renormalize to sum of scores including ISR')
    parser.add_argument('--jet-assign-choice', type=int, default=2,
                    help='0: category with max value \
                          1: no more than 1 per lep, 3 per had \
                          2: at least 1 lep, 2 had \
                          3: at least 2 lep, 2 had \
                          4: at least 2 lep, 3 had')
    parser.add_argument('--mass-reco-choice', type=int, default=0,
                    help='0: min mass \
                          1: max mass \
                          2: min pt mass \
                          3: max pt mass')

    return analysis_name, analysis_defaults


class Eval2HDM():

    def __init__(self, args):
        #super().__init__(args)
        self.bm = BatchManager2HDM(args)


    def run(self):
        args = self.args
        test_sample = args.test_sample if args.test_sample >= 0 else self.bm.get_nr_events()//10
        output_dir = self.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        param_file = output_dir / 'NN.pt'
        if os.path.exists(param_file):
            model = torch.load(param_file)
        else:
            log.error("Running without training but the model file {} is not present".format(param_file))
        model.eval()
        test_torch_batch = bm.get_torch_batch(test_sample)
        X, Y = train_torch_batch[0], train_torch_batch[1]            
        P = model(X)
        Y = Y.reshape(-1, args.totaloutputs)

        mass, mass_perfect = mass_reco(args, "Test")
        mass_t, mass_perfect_t = mass_reco(args, "6-jet")

        fig = plt.figure()
        bins_m = np.linspace(0, 800, 100)
        plt.hist(mass, bins=bins_m, histtype='step',
                 label='Mass, reco=%d'%args.mass_reco_choice, density=True)
        plt.hist(mass_perfect, bins=bins_m, histtype='step',
                 label='Mass perfect, fraction=%f'%(len(mass_perfect)/len(mass)), density=True)
        plt.hist(mass_t, bins=bins_m, histtype='step',
                 label='Mass, test', density=True)
        plt.xlabel('$S$ mass [GeV]')
        plt.ylim(0,0.012 if args.small else 0.03) 
        plt.legend()
        plt.grid()
        title = 'reco_mass_choice%d_assign%d'%(args.mass_reco_choice, args.jet_assign_choice)
        if args.allow_2jet:
            title += '_allow2jet'
        #if args.exclude_highest_ISR:
        #    title += '_excludehighestISR'
        if args.normalize_scores:
            title += '_normalizescores'
        if args.normalize_scores_ISR:
            title += '_normalizescoresISR'
        if args.small:
            title += '_small'
        plt.savefig(args.output_dir / (title+'.png') )


    def minv(a):
        return np.sqrt(np.maximum(np.zeros(len(a)),a[:,3]**2 - a[:,2]**2 - a[:,1]**2 - a[:,0]**2))
    def pt(a):
        return np.sqrt(a[:,1]**2 + a[:,0]**2)

    def mass_reco(args, sample):

        data = np.load(args.npz)
        jets = data["jets_{}".format(sample)]*1e-3 # Convert to GeV
        ISR_pred_score = data["pred_ISR_{}".format(sample)]
        ISR_pred_score = np.ones_like(ISR_pred_score) - ISR_pred_score
        had0_pred_score = data["pred_had0top_{}".format(sample)]
        lep0_pred_score = data["pred_lep0top_{}".format(sample)]
        lep1_pred_score = data["pred_lep1top_{}".format(sample)]
        ISR_truth = data["truth_ISR_{}".format(sample)]
        had0_truth = data["truth_had0top_{}".format(sample)]
        lep0_truth = data["truth_lep0top_{}".format(sample)]
        lep1_truth = data["truth_lep1top_{}".format(sample)]

        nevents = len(jets)
        print("Npz file has ",nevents,sample)

        cut = jets[:,:,3].min(axis=1)>=0 #right now not cutting at all
        jets = jets[cut]
        ISR_pred_score = ISR_pred_score[cut]
        had0_pred_score = had0_pred_score[cut]
        lep0_pred_score = lep0_pred_score[cut]
        lep1_pred_score = lep1_pred_score[cut]
        nevents = len(jets)
        print("Npz file after cut has ",nevents)

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

        #print(ISR_pred_score [:1])
        #print(had0_pred_score[:1])
        #print(lep0_pred_score[:1])
        #print(lep1_pred_score[:1])
        #print(ISR_truth [:1])
        #print(had0_truth[:1])
        #print(lep0_truth[:1])
        #print(lep1_truth[:1])

        
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

        perfect = np.logical_xor(had0_truth, had0_choice) | np.logical_xor(lep0_truth, lep0_choice) | np.logical_xor(lep1_truth, lep1_choice)
        perfect = perfect.sum(1)==0
        #print(had0_truth[1])
        #print(had0_choice[1])
        #print(had0_pred_score[1])
        #print(lep0_truth[1])
        #print(lep0_choice[1])
        #print(lep0_pred_score[1])
        #print(lep1_truth[1])
        #print(lep1_choice[1])
        #print(lep1_pred_score[1])
        choice0 = np.concatenate([np.tile([1,0,0],(len(had0_choice),1)), had0_choice | lep0_choice], axis=1)
        choice1 = np.concatenate([np.tile([0,1,0],(len(had0_choice),1)), had0_choice | lep1_choice], axis=1)

        print("On average lep0 jets:",lep0_choice.sum(1).mean())
        print("On average lep1 jets:",lep1_choice.sum(1).mean())
        print("On average had0 jets:",had0_choice.sum(1).mean())
        print("Perfect fraction",perfect.mean())

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

        return mass, mass[perfect]
