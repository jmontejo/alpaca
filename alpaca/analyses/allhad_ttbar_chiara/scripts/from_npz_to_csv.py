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

def tj_distr(arg):
    data = np.load(args.npz)
    mydict={}
    for i in range(7):
        # print(i)
        mydict['jet_px_'+str(i)] = data['jets'][:,i,0]
        mydict['jet_py_'+str(i)] = data['jets'][:,i,1]
        mydict['jet_pz_'+str(i)] = data['jets'][:,i,2]
        mydict['jet_e_'+str(i)] = data['jets'][:,i,3]
        mydict['from_top_'+str(i)] = data['pred_ISR'][:,i]
        mydict['from_top_'+str(i)+'_true'] = data['truth_ISR'][:,i]

    for i in range(5):
        mydict['same_as_lead_'+str(i)] = data['pred_decay'][:,i]
        mydict['same_as_lead_'+str(i)+'_true'] = data['truth_decay'][:,i]

    df= pd.DataFrame.from_dict(mydict)

    jets = data["jets"]*1e-3 # Convert to GeV
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
    # ISR_truth = data["truth_ISR"]
    # ISR_truth = ISR_truth[cut]
    # ttbar_truth = data["truth_decay"]
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

    minv_top1_pred, minv_top2_pred  = tj_fixedcount_alg(data,jets, cut)
    df['m1_riccardo'] = minv_top1_pred
    df['m2_riccardo'] = minv_top2_pred
    df.to_csv(args.csv)
        
    return minv_top1_pred


def tj_fixedcount_alg(data,jets,cut):

    njets = jets.shape[1]
    # Plot network-labeled jet mass distribution
    ISR_pred_score = data["pred_ISR"]
    ISR_pred_score = ISR_pred_score[cut]
    ttbar_pred_score = data["pred_decay"]
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

    top_had1 = jets_top1_pred.sum(1)
    top_had2 = jets_top2_pred.sum(1)

    minv_top1_pred = np.where(pt(top_had1) >= pt(top_had2), minv(top_had1), minv(top_had2))
    minv_top2_pred = np.where(pt(top_had1) <  pt(top_had2), minv(top_had1), minv(top_had2))
        
    return minv_top1_pred, minv_top2_pred

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Build csv.')
    parser.add_argument('--npz', type=Path,
                        help='path to the npz file out of alpaca', default='/afs/cern.ch/user/j/jmontejo/public/ForRiccardo/hydra.npz')
    parser.add_argument('--csv', default='hydra_riccardo.csv',
                        help='output csv')
    args = parser.parse_args()
    tj_top1 = tj_distr(args)
