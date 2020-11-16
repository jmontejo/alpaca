#!/usr/bin/env python

# from  npz files (Riccardo's, Javier's, ...) to csv in the same format as mine
# very hardcoded, meant for a quick validation of the m(top) distribution

import pandas as pd
import uproot
import awkward

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def minv(a):
    return np.sqrt(a[:,3]**2 - a[:,2]**2 - a[:,1]**2 - a[:,0]**2)

def pt(a):
    return np.sqrt(a[:,1]**2 + a[:,0]**2)

from functools import lru_cache

def arrayM(a):
    from math import sqrt
    m = sqrt(-a[0]**2-a[1]**2-a[2]**2+a[3]**2)
    assert m>0, a
    return m

def arrayPt(a):
    from math import sqrt
    return sqrt(a[0]**2+a[1]**2)

def tj_distr(args, sample=None):
    data = np.load(args.npz)

    nnd = {item: data[item] for item in data.files}
    #print(mydict)
    for k in nnd:
        print(k)
        # print(nnd[k].shape)
        '''
        jets_Test
        (5000, 7, 4)
        pred_ISR_Test
        (5000, 7)
        truth_ISR_Test
        (5000, 7)
        pred_ttbar_Test
        (5000, 5)
        truth_ttbar_Test
        (5000, 5)
        pred_bjet_Test
        (5000, 6)
        truth_bjet_Test
        (5000, 6)
        '''
    if not sample:
        sample = "Test"

    mydict={}
    for i in range(7):
        # print(i)
        mydict['jet_px_'+str(i)] = nnd['jets_'+sample][:,i,0]
        mydict['jet_py_'+str(i)] = nnd['jets_'+sample][:,i,1]
        mydict['jet_pz_'+str(i)] = nnd['jets_'+sample][:,i,2]
        mydict['jet_e_'+str(i)] = nnd['jets_'+sample][:,i,3]
        mydict['from_top_'+str(i)] = nnd['pred_ISR_'+sample][:,i]
        mydict['from_top_'+str(i)+'_true'] = nnd['truth_ISR_'+sample][:,i]
    #for i in range(6):
    #    mydict['is_b_'+str(i)] = nnd['pred_bjet_'+sample][:,i]
    #    mydict['is_b_'+str(i)+'_true'] = nnd['truth_bjet_'+sample][:,i]
    for i in range(5):
        mydict['same_as_lead_'+str(i)] = nnd['pred_ttbar_'+sample][:,i]
        mydict['same_as_lead_'+str(i)+'_true'] = nnd['truth_ttbar_'+sample][:,i]

    #for k in mydict:
    #    print(k)
    #print(mydict)

    print('\n\n\n')
    df= pd.DataFrame.from_dict(mydict)
    # print('df columns')
    # print(df.columns)
    # print('df index')
    # print(df.index)
    # print('df shape')
    # print(df.shape)
    # print('df head')
    # print(df.head())
    
    #print('\n\n\n')

    jets = data["jets_{}".format(sample)]*1e-3 # Convert to GeV
    cut = jets[:,:,3].min(axis=1)>=0 #right now not cutting at all
    jets = jets[cut]
    nevents = len(jets)
    print("Npz file has ",nevents)

    #cut = cut & lead10.astype(np.bool)
    jets = jets[cut]
    nevents = len(jets)
    print("Npz file after cut has ",nevents)

    njets = jets.shape[1]

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

    # minv_top1_pred = my_varcount_alg(data,jets,sample, cut, args)
    minv_top1_pred, minv_top2_pred  = tj_fixedcount_alg(data,jets,sample, cut)
    df['m1_javier'] = minv_top1_pred
    df['m2_javier'] = minv_top2_pred
    print('hey chiara!')
    print('minv_top1_pred sum', minv_top1_pred.sum())
    print('minv_top2_pred sum', minv_top2_pred.sum())
    # print('minv_top1_pred')
    # print(minv_top1_pred)
    # name_csv = str(args.npz)
    # name_csv = name_csv.replace('.npz','_'+sample+'.csv')
    name_csv='hydra_javier_'+sample+'.csv'
    df.to_csv(name_csv)
        
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

    pt_top1 = pt(jets_top1_pred.sum(1))
    pt_top2 = pt(jets_top2_pred.sum(1))

    if sample == 'Test':
        print('hey chiara!')
        print('minv_top1_pred')
        print(minv_top1_pred)
        print('minv_top2_pred')
        print(minv_top2_pred)
        print('pt_top1')
        print(pt_top1)
        print('pt_top2')
        print(pt_top2)
        print('inverted')
        print((pt_top1<pt_top2).sum())
        
    return minv_top1_pred, minv_top2_pred

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Compare masses.')
    parser.add_argument('--npz', type=Path,
                        help='path to the npz file out of alpaca', default='/afs/cern.ch/user/j/jmontejo/public/ForRiccardo/hydra.npz')
    parser.add_argument('--allow-2jet', action="store_true",
                        help='Use two or more jets for the tops')
    parser.add_argument('--normalize-scores', action="store_true",
                        help='Renormalize to sum of scores')
    parser.add_argument('--normalize-scores-ISR', action="store_true",
                        help='Renormalize to sum of scores including ISR')
    parser.add_argument('--exclude-highest-ISR', action="store_true")
    parser.add_argument('--small', action="store_true")
    parser.add_argument('--output-dir', type=Path, default='plots/',
                        help='path to the output directory')
    args = parser.parse_args()


    tj_top1 = tj_distr(args, sample="Test")
    tj_top1_nopartonsel = tj_distr(args, sample="6-jet")

    fig = plt.figure()
    bins_chi = np.linspace(0, 40, 400)
    bins_chi_coarse = np.linspace(0, 40, 40)

    fig = plt.figure()
    bins_m = np.linspace(0, 500, 100)
    if not args.small:
        plt.hist(tj_top1, bins=bins_m, histtype='step', label='Top 1 (NN pred) all partons in sample',
                 density=True)
    plt.xlabel('$t_1$ mass [GeV]')
    #plt.ylim(0,0.012 if args.small else 0.03) 
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
