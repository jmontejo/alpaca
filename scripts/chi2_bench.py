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

def tj_distr(npz_file, dochi2=False, sample=None):
    data = np.load(npz_file)

    if not sample:
        sample = "6-jet"
    jets = data["jets_{}".format(sample)]*1e-3 # Convert to GeV
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
    if dochi2:
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

    # Plot network-labeled jet mass distribution
    ISR_pred_score = data["pred_ISR_{}".format(sample)]
    ISR_pred_score = ISR_pred_score[cut]
    ttbar_pred_score = data["pred_ttbar_{}".format(sample)]
    ttbar_pred_score = ttbar_pred_score[cut]

    # Convert scores to flags
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
    return minv_top1_pred, chi2, mass

    # for i in range(10):
    #     print('\n\n',i)
    #     print(jets[i])
    #     print(ISR_truth[i])
    #     print(ISR_pred[i])
    #     print(ttbar_truth[i])
    #     print(ttbar_pred[i])

    #histargs = {"bins":50, "range":(0.,250.), "density":False, "histtype":'step'}

    #fig = plt.figure()
    #plt.hist(minv_top1_truth, label='Top 1 (truth)', **histargs, alpha=0.5, fill=True)
    #plt.hist(minv_top2_truth, label='Top 2 (truth)', **histargs, alpha=0.5, fill=True)
    #plt.hist(minv_top1_pred, label='Top 1 (pred)', **histargs)
    #plt.hist(minv_top2_pred, label='Top 2 (pred)', **histargs)
    #plt.legend(loc="upper left")
    #plt.savefig('{}/top_mass_{}.png'.format(datadir,sample))
    #plt.close()

    #perfectISR = np.equal(ISR_pred,ISR_truth).sum(1)==njets
    #print("Perfect ISR fraction ({}) = {} / {} = {:.3f}".format(sample,perfectISR.sum(),nevents,float(perfectISR.sum())/nevents))
    #perfectmatch = perfectISR & (np.equal(ttbar_pred,ttbar_truth).sum(1)==6)
    #print("Perfect match fraction ({}) = {} / {} = {:.3f}".format(sample,perfectmatch.sum(),nevents,float(perfectmatch.sum())/nevents))
    #print("\n")


if __name__ == '__main__':

    # jets_etaphi = [[ 97242.921 , 0.6877870 , -1.628724 , 121789.06],
    #         [ 91254.734 , -0.077293 , 2.6106293 , 91859.867],
    #         [ 86505.171 , -1.494373 , 0.4952870 , 202573.29],
    #         [ 71843.148 , 0.2761235 , -2.958802 , 75055.796],
    #         [     71331 , -0.861143 , 1.8417986 ,     99926],
    #         [ 60377.855 , -1.335611 , -0.644344 , 122985.21],
    #        ]
    # jets_etaphi = np.ndarray((6,4),buffer=np.array(jets_etaphi))
    # jets = np.copy(jets_etaphi)
    # jets[:,0] = jets_etaphi[:,0]*np.cos(jets_etaphi[:,2])
    # jets[:,1] = jets_etaphi[:,0]*np.sin(jets_etaphi[:,2])
    # jets[:,2] = jets_etaphi[:,0]*np.sinh(jets_etaphi[:,1])
    # jets = jets*1e-3
    # tripletsNoBtag =  generateTripletsNoBtag(jets)
    # results = evalPerm(tripletsNoBtag)
    # print(results)
    # sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description='Chi2 benchmark.')
    parser.add_argument('--xsttbar-dir', required=True, type=Path,
                        help=('path to the dir where you have downloaded the '
                              'dataset'))
    parser.add_argument('--npz', type=Path,
                        help='path to the npz file out of alpaca')
    parser.add_argument('--do-chi2', action="store_true",
                    help='Compute chi2 and compare against it')
    parser.add_argument('--output-dir', type=Path,
                        help='path to the output directory')
    args = parser.parse_args()

    dataset_nobfixed = 'user.rpoggi.410471.PhPy8EG.DAOD_TOPQ1.e6337_e5984_s3126_r9364_r9315_p3629.TTDIFFXS36_R21_allhad_resolved.root'
    dataset_bfixed = 'user.rpoggi.410471.PhPy8EG.DAOD_TOPQ1.e6337_e5984_s3126_r9364_r9315_p3629.TTDIFFXS34_R21_allhad_resolved.root'

    no_bfixed = get_base_events(list((args.xsttbar_dir / dataset_nobfixed).glob('*.root')))
    bfixed = get_base_events(list((args.xsttbar_dir / dataset_bfixed).glob('*.root')))


    tj_top1, unused1, unused2 = tj_distr(args.npz, sample="Test")
    tj_top1_notallp, chi2, chi2mass = tj_distr(args.npz, args.do_chi2, sample="6-jet")

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


    no_bfixed = no_bfixed[no_bfixed['reco_Chi2Fitted'] < 10]
    bfixed = bfixed[bfixed['reco_Chi2Fitted'] < 10]
    chi2mass_skim = chi2mass[chi2<10]

    fig = plt.figure()
    bins_m = np.linspace(0, 500, 100)
    plt.hist(no_bfixed['reco_t1_m'] / 1000, bins=bins_m, histtype='step',
             label='$\chi^2 < 10$ no bfixed', density=True)
    plt.hist(bfixed['reco_t1_m'] / 1000, bins=bins_m, histtype='step',
             label='$\chi^2 < 10$ bfixed', density=True)
    plt.hist(tj_top1, bins=bins_m, histtype='step', label='Top 1 (NN pred) all partons in sample',
             density=True)
    plt.hist(tj_top1_notallp, bins=bins_m, histtype='step', label='Top 1 (NN pred) no parton cut',
             density=True)
    if args.do_chi2:
        plt.hist(chi2mass_skim, bins=bins_m, histtype='step', label='Top 1 (my chi2 no bfixed < 10) no parton cut',
             density=True)
    plt.xlabel('$t_1$ mass [GeV]')
    plt.ylim(0,0.03) 
    plt.legend()
    plt.grid()
    if args.do_chi2:
        plt.savefig(args.output_dir / 'mass_dist_withmychi2.png')
    else:
        plt.savefig(args.output_dir / 'mass_dist.png')

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
