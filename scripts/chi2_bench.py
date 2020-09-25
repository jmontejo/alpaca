#!/usr/bin/env python

import uproot
import awkward

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def get_base_events(input_files):
    nominal = uproot.lazyarrays(
        input_files,
        'nominal',
        ['eventNumber', 'reco_Chi2Fitted', 'reco_t1_m', 'reco_t2_m', 'jet_*'],
    )

    sel = nominal
    sel = sel[sel['jet_pt'].counts >= 6]
    #sel = sel[sel['jet_pt'].counts <= 7]
    sel = sel[sel['jet_pt'].min() > 25000.]
    return sel

def minv(a):
    return np.sqrt(a[:,3]**2 - a[:,2]**2 - a[:,1]**2 - a[:,0]**2)
def pt(a):
    return np.sqrt(a[:,1]**2 + a[:,0]**2)

from functools import lru_cache

@lru_cache(maxsize=32)
def generatePerms(n):
    from itertools import permutations,combinations
    sixes = list(combinations(range(n),6))
    triplets = []
    for six in sixes:
        bpairs = list(combinations(six,2))
        for bpair in bpairs:
            lights = [ jet for jet in six if jet not in bpair ]
            triplet1 = (bpair[0],lights[0],lights[1])
            triplet2 = (bpair[1],lights[2],lights[3])
            triplets.append((triplet1,triplet2))
            triplet1 = (bpair[0],lights[2],lights[3])
            triplet2 = (bpair[1],lights[0],lights[1])
            triplets.append((triplet1,triplet2))
            triplet1 = (bpair[0],lights[0],lights[3])
            triplet2 = (bpair[1],lights[1],lights[2])
            triplets.append((triplet1,triplet2))
            triplet1 = (bpair[0],lights[1],lights[2])
            triplet2 = (bpair[1],lights[0],lights[3])
            triplets.append((triplet1,triplet2))
            triplet1 = (bpair[0],lights[0],lights[2])
            triplet2 = (bpair[1],lights[1],lights[3])
            triplets.append((triplet1,triplet2))
            triplet1 = (bpair[0],lights[1],lights[3])
            triplet2 = (bpair[1],lights[0],lights[2])
            triplets.append((triplet1,triplet2))
    return triplets

def generateTripletsNoBtag(jetlist):
    while jetlist[-1,3]==0: jetlist = jetlist[:-1]
    perms = generatePerms(len(jetlist))
    #triplets = []
    for p in perms:
        yield ((jetlist[p[0][0]], jetlist[p[0][1]], jetlist[p[0][2]]),(jetlist[p[1][0]], jetlist[p[1][1]], jetlist[p[1][2]] ))
        #triplets.append( [jetlist[x] for sublist in p for x in sublist] )
    #return triplets

def arrayM(a):
    from math import sqrt
    m = sqrt(-a[0]**2-a[1]**2-a[2]**2+a[3]**2)
    assert m>0, a
    return m

def arrayPt(a):
    from math import sqrt
    return sqrt(a[0]**2+a[1]**2)


def evalChi2(triplet1,triplet2):
    #sigma_mbjj = 10.7 #check also factor 2 below
    #sigma_mjj = 5.9
    sigma_mbjj = 17.6
    sigma_mjj = 9.3
    mW = 80.4

    mjj1 = arrayM((triplet1[1]+triplet1[2]))
    mjj2 = arrayM((triplet2[1]+triplet2[2]))

    mbjj1 = arrayM((triplet1[0]+triplet1[1]+triplet1[2]))
    mbjj2 = arrayM((triplet2[0]+triplet2[1]+triplet2[2]))

    dmbjj = (mbjj1-mbjj2)
    dmjj1 = (mjj1-mW)
    dmjj2 = (mjj2-mW)

    chi2 = dmbjj*dmbjj/(2*sigma_mbjj*sigma_mbjj) + dmjj1*dmjj1/(sigma_mjj*sigma_mjj) + dmjj2*dmjj2/(sigma_mjj*sigma_mjj)
    return chi2

def evalPerm(triplets):
    chi2min = 1e12
    for triplet1,triplet2 in triplets:
        chi2 = evalChi2(triplet1,triplet2)
        if chi2<chi2min:
            chi2min = chi2
            #
            top1 = (triplet1[0]+triplet1[1]+triplet1[2])
            top2 = (triplet2[0]+triplet2[1]+triplet2[2])
            b1 = triplet1[0]
            b2 = triplet2[0]
            W1 = (triplet1[1]+triplet1[2])
            W2 = (triplet2[1]+triplet2[2])
            #
            if arrayPt(top1)>arrayPt(top2):
                besttop1 = top1
                bestW1 = W1
                besttop2 = top2
                bestW2 = W2
            else:
                besttop1 = top2
                bestW1 = W2
                besttop2 = top1
    return np.array([chi2min, arrayM(besttop1)])

def eval_chi2(jets):
    tripletsNoBtag =  np.array(list(map(generateTripletsNoBtag,jets)))
    results = np.vstack(np.array(list(map(evalPerm,tripletsNoBtag))))
    return results[:,0], results[:,1]

def tj_distr(args, sample=None):
    data = np.load(args.npz)

    if not sample:
        sample = "6-jet"
    jets = data["jets_{}".format(sample)]*1e-3 # Convert to GeV
    cut = jets[:,:,3].min(axis=1)>=0 #right now not cutting at all
    jets = jets[cut]
    nevents = len(jets)
    print("Npz file has ",nevents)

    #lead10 = np.concatenate([np.ones(4),np.zeros(nevents-4)])
    #print(lead10[:12])
    cut = jets[:,:,3].min(axis=1)>=0 #can cut on exactly 6 jets using ==0
    #    if not (jets[5].Pt()*GeV>55): continue

    #cut = cut & lead10.astype(np.bool)
    jets = jets[cut]
    nevents = len(jets)
    print("Npz file after cut has ",nevents)

    njets = jets.shape[1]
    if args.do_chi2:
        chi2, mass = eval_chi2(jets)
    else:
        chi2 = np.zeros(nevents)
        mass = np.zeros(nevents)

    # # Plot truth jet mass distribution
    # ISR_truth = data["truth_ISR_{}".format(sample)]
    # ISR_truth = ISR_truth[cut]
    # ttbar_truth = data["truth_ttbar_{}".format(sample)]
    # ttbar_truth = ttbar_truth[cut]
    # # Add a 1 for the leading jet
    # ttbar_truth = np.concatenate([np.ones([nevents,1]),ttbar_truth],1)
    # # Mask out ISR jets
    # jets_ttbar_truth = jets[ISR_truth.astype(np.bool)].reshape(nevents,6,4)
    # # Mask out the jets from each of the tops
    # jets_top1_truth = jets_ttbar_truth[ttbar_truth.astype(np.bool)].reshape(nevents,3,4)
    # jets_top2_truth = jets_ttbar_truth[~ttbar_truth.astype(np.bool)].reshape(nevents,3,4)
    # minv_top1_truth = minv(jets_top1_truth.sum(1))
    # minv_top2_truth = minv(jets_top2_truth.sum(1))

    minv_top1_pred = my_varcount_alg(data,jets,sample, cut, args)
    #minv_top1_pred = tj_fixedcount_alg(data,jets,sample, cut)

    return minv_top1_pred, chi2, mass

def my_varcount_alg(data,jets,sample,cut, args):

    # Plot network-labeled jet mass distribution
    ISR_pred_score = data["pred_ISR_{}".format(sample)]
    ISR_pred_score = ISR_pred_score[cut]
    had1_pred_score = data["pred_ttbar_{}".format(sample)]
    had1_pred_score = had1_pred_score[cut]
    had2_pred_score = data["pred_lep1top_{}".format(sample)]
    had2_pred_score = had2_pred_score[cut]
    if args.normalize_scores:
        sum_score = had1_pred_score+had2_pred_score
    elif args.normalize_scores_ISR:
        sum_score = had1_pred_score+had2_pred_score+ISR_pred_score
    else:
        sum_score = 1
    had1_pred_score /= sum_score
    had2_pred_score /= sum_score
    ISR_pred_score /= sum_score


    # Convert scores to flags
    nevents = len(had1_pred_score)
    had1_choice = np.zeros(had1_pred_score.shape)
    _had1_pred_score = np.array(had1_pred_score) # temp var to modify
    had2_choice = np.zeros(had2_pred_score.shape)
    _had2_pred_score = np.array(had2_pred_score) # temp var to modify

    if args.exclude_highest_ISR:
        _had2_pred_score[np.arange(nevents), ISR_pred_score.argmax(1)] = 0
        _had1_pred_score[np.arange(nevents), ISR_pred_score.argmax(1)] = 0
    usenjet = 2 if args.allow_2jet else 3
    for i in range(usenjet): #choose two jets for one top
        # choose the highest scoring jet
        had1_choice[np.arange(nevents), _had1_pred_score.argmax(1)] = 1
        # set the score to 0 to ignore it in the next iteration
        _had2_pred_score[np.arange(nevents), _had1_pred_score.argmax(1)] = 0
        _had1_pred_score[np.arange(nevents), _had1_pred_score.argmax(1)] = 0

        # choose the highest scoring jet
        had2_choice[np.arange(nevents), _had2_pred_score.argmax(1)] = 1
        # set the score to 0 to ignore it in the next iteration
        _had1_pred_score[np.arange(nevents), _had2_pred_score.argmax(1)] = 0
        _had2_pred_score[np.arange(nevents), _had2_pred_score.argmax(1)] = 0

    threshold = 0.5 if args.allow_2jet else 1.5
    had1_above_threshold = _had1_pred_score.max(1) > threshold
    had1_choice[np.arange(nevents), _had1_pred_score.argmax(1)] = had1_above_threshold
    had2_above_threshold = _had2_pred_score.max(1) > threshold
    had2_choice[np.arange(nevents), _had2_pred_score.argmax(1)] = had2_above_threshold

    jets_had1 = np.copy(jets)
    jets_had2 = np.copy(jets)
    jets_had1[~had1_choice.astype(np.bool)] = 0
    jets_had2[~had2_choice.astype(np.bool)] = 0

    top_had1 = jets_had1.sum(1)
    top_had2 = jets_had2.sum(1)

    minv_top1_pred = np.where(pt(top_had1) >= pt(top_had2), minv(top_had1), minv(top_had2))
    minv_top2_pred = np.where(pt(top_had1) <  pt(top_had2), minv(top_had1), minv(top_had2))

    return minv_top1_pred

def tj_fixedcount_alg(data,jets,sample,cut):

    njets = jets.shape[1]
    # Plot network-labeled jet mass distribution
    ISR_pred_score = data["pred_ISR_{}".format(sample)]
    ISR_pred_score = ISR_pred_score[cut]
    ttbar_pred_score = data["pred_ttbar_{}".format(sample)]
    ttbar_pred_score = ttbar_pred_score[cut]

    # Convert scores to flags
    nevents = len(ISR_pred_score)
    ISR_pred = np.ones(ISR_pred_score.shape)
    ttbar_pred = np.zeros([nevents,6])
    ttbar_pred[:,0] = 1
    _ISR_pred_score = np.array(ISR_pred_score) # temp var to modify
    for i in range(6,njets):
        # remove the lowest scoring jet
        ISR_pred[np.arange(nevents), _ISR_pred_score.argmin(1)] = 0
        # set the min score to 1
        _ISR_pred_score[np.arange(nevents), _ISR_pred_score.argmin(1)] = 1
    #
    _ttbar_pred_score = np.array(ttbar_pred_score) # temp var to modify
    for i in range(2):
        # add the highest scoring jet
        ttbar_pred[np.arange(nevents), _ttbar_pred_score.argmax(1)+1] = 1
        # zero out the max score
        _ttbar_pred_score[np.arange(nevents), _ttbar_pred_score.argmax(1)] = 0
    #
    jets_ttbar_pred = jets[ISR_pred.astype(np.bool)].reshape(nevents,6,4)
    jets_top1_pred = jets_ttbar_pred[ttbar_pred.astype(np.bool)].reshape(nevents,3,4)
    jets_top2_pred = jets_ttbar_pred[~ttbar_pred.astype(np.bool)].reshape(nevents,3,4)
    minv_top1_pred = minv(jets_top1_pred.sum(1))
    minv_top2_pred = minv(jets_top2_pred.sum(1))
    return minv_top1_pred

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Chi2 benchmark.')
    parser.add_argument('--xsttbar-dir', required=True, type=Path,
                        help=('path to the dir where you have downloaded the '
                              'dataset'))
    parser.add_argument('--npz', type=Path,
                        help='path to the npz file out of alpaca')
    parser.add_argument('--do-chi2', action="store_true",
                    help='Compute chi2 and compare against it')
    parser.add_argument('--allow-2jet', action="store_true",
                    help='Use two or more jets for the tops')
    parser.add_argument('--normalize-scores', action="store_true",
                    help='Renormalize to sum of scores')
    parser.add_argument('--normalize-scores-ISR', action="store_true",
                    help='Renormalize to sum of scores including ISR')
    parser.add_argument('--exclude-highest-ISR', action="store_true")
    parser.add_argument('--small', action="store_true")
    parser.add_argument('--output-dir', type=Path,
                        help='path to the output directory')
    args = parser.parse_args()

    dataset_nobfixed = 'user.rpoggi.410471.PhPy8EG.DAOD_TOPQ1.e6337_e5984_s3126_r9364_r9315_p3629.TTDIFFXS36_R21_allhad_resolved.root'
    dataset_bfixed = 'user.rpoggi.410471.PhPy8EG.DAOD_TOPQ1.e6337_e5984_s3126_r9364_r9315_p3629.TTDIFFXS34_R21_allhad_resolved.root'

    no_bfixed = get_base_events(list((args.xsttbar_dir / dataset_nobfixed).glob('*.root')))
    bfixed = get_base_events(list((args.xsttbar_dir / dataset_bfixed).glob('*.root')))


    tj_top1, unused1, unused2 = tj_distr(args, sample="Test")
    tj_top1_notallp, chi2, chi2mass = tj_distr(args, sample="6-jet")

    fig = plt.figure()
    bins_chi = np.linspace(0, 40, 400)
    bins_chi_coarse = np.linspace(0, 40, 40)

    plt.hist(no_bfixed['reco_Chi2Fitted'], bins=bins_chi, histtype='step',
             label='$\chi^2$ no bfixed')
    plt.hist(bfixed['reco_Chi2Fitted'], bins=bins_chi, histtype='step',
             label='$\chi^2$ bfixed')
    if args.do_chi2:
        factor = len(bfixed['reco_Chi2Fitted'])/len(chi2)/10
        print("My chi2 scale factor",factor)
        plt.hist(chi2, bins=bins_chi_coarse, histtype='step',
             label='$\chi^2$ mine', weights=np.ones_like(chi2)*factor)
    plt.legend()
    plt.grid()
    if args.do_chi2:
        plt.savefig(args.output_dir / 'chi2_dist_withmychi2.png')
    else:
        plt.savefig(args.output_dir / 'chi2_dist.png')


    no_bfixed_skim = no_bfixed[no_bfixed['reco_Chi2Fitted'] < 10]
    bfixed_skim = bfixed[bfixed['reco_Chi2Fitted'] < 10]
    chi2mass_skim = chi2mass[chi2<10]
    tj_top1_notallp_skim = tj_top1_notallp[chi2<10]

    fig = plt.figure()
    bins_m = np.linspace(0, 500, 100)
    if not args.small:
        plt.hist(no_bfixed_skim['reco_t1_m'] / 1000, bins=bins_m, histtype='step',
                 label='$\chi^2 < 10$ no bfixed', density=True)
        plt.hist(bfixed_skim['reco_t1_m'] / 1000, bins=bins_m, histtype='step',
                 label='$\chi^2 < 10$ bfixed', density=True)
        plt.hist(tj_top1, bins=bins_m, histtype='step', label='Top 1 (NN pred) all partons in sample',
                 density=True)
    plt.hist(tj_top1_notallp, bins=bins_m, histtype='step', label='Top 1 (NN pred) no parton cut',
             density=True)
    if args.do_chi2:
        plt.hist(chi2mass_skim, bins=bins_m, histtype='step', label='Top 1 (my chi2 no bfixed < 10) no parton cut',
             density=True)
        plt.hist(tj_top1_notallp_skim, bins=bins_m, histtype='step', label='Top 1 (NN pred, my chi2 < 10) no parton cut',
             density=True)
    plt.hist(no_bfixed['reco_t1_m'] / 1000, bins=bins_m, histtype='step',
             label='$\chi^2$ no cut', density=True)
    plt.xlabel('$t_1$ mass [GeV]')
    plt.ylim(0,0.012 if args.small else 0.03) 
    plt.legend()
    plt.grid()
    title = 'mass_dist'
    if args.allow_2jet:
        title += '_allow2jet'
    if args.exclude_highest_ISR:
        title += '_excludehighestISR'
    if args.normalize_scores:
        title += '_normalizescores'
    if args.normalize_scores_ISR:
        title += '_normalizescoresISR'
    if args.do_chi2:
        title += '_withmychi2'
    if args.small:
        title += '_small'
    plt.savefig(args.output_dir / (title+'.png') )

    #truth = uproot.lazyarrays(
    #    input_files,
    #    'truth',
    #    ['eventNumber', 'MC_b_from_t*', 'MC_Wdecay*_from_t*'],
    #)

    #print('converting nominal')
    #nominal_df = awkward.topandas(nominal)
    #print(nominal_df)
    #truth_df = awkward.topandas(truth)
    #print(truth_df)

    #import sys
    #sys.exit()

    #selected = data[data['reco_Chi2Fitted'] < 10]


    #fig = plt.figure()
    #bins_m = np.linspace(0, 500, 100)
    #plt.hist(data['reco_t1_m'] / 1000., bins=bins_m, histtype='step',
    #         label='no $\chi^2$ cut')
    #plt.hist(selected['reco_t1_m'] / 1000., bins=bins_m, histtype='step',
    #         label='$\chi^2 < 10$')
    #plt.xlabel('Mass [GeV]')
    #plt.legend()
    #plt.grid()
    #plt.show()
