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
    sel = sel[sel['jet_pt'].counts == 6]
    sel = sel[sel['jet_pt'].min() > 25000.]
    return sel

def minv(a):
    return np.sqrt(a[:,3]**2 - a[:,2]**2 - a[:,1]**2 - a[:,0]**2)


def tj_distr(npz_file):
  data = np.load(npz_file)

  for sample in ["Test","Train"]:#,"6-jet"]:
    sample = 'Test'
    jets = data["jets_{}".format(sample)]*1e-3 # Convert to GeV
    nevents = len(jets)
    njets = jets.shape[1]

    # Plot truth jet mass distribution
    ISR_truth = data["truth_ISR_{}".format(sample)]
    # Add a 1 for the leading jet
    ttbar_truth = np.concatenate([np.ones([nevents,1]),data["truth_ttbar_{}".format(sample)]],1)
    # Mask out ISR jets
    jets_ttbar_truth = jets[ISR_truth.astype(np.bool)].reshape(nevents,6,4)
    # Mask out the jets from each of the tops
    jets_top1_truth = jets_ttbar_truth[ttbar_truth.astype(np.bool)].reshape(nevents,3,4)
    jets_top2_truth = jets_ttbar_truth[~ttbar_truth.astype(np.bool)].reshape(nevents,3,4)
    minv_top1_truth = minv(jets_top1_truth.sum(1))
    minv_top2_truth = minv(jets_top2_truth.sum(1))

    # Plot network-labeled jet mass distribution
    ISR_pred_score = data["pred_ISR_{}".format(sample)]
    ttbar_pred_score = data["pred_ttbar_{}".format(sample)]
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
    yield minv_top1_pred

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
    import argparse
    parser = argparse.ArgumentParser(description='Chi2 benchmark.')
    parser.add_argument('--xsttbar-dir', required=True, type=Path,
                        help=('path to the dir where you have downloaded the '
                              'dataset'))
    parser.add_argument('--npz', type=Path,
                        help='path to the npz file out of alpaca')
    args = parser.parse_args()

    #dataset_nobfixed = 'user.rpoggi.410471.PhPy8EG.DAOD_TOPQ1.e6337_e5984_s3126_r9364_r9315_p3629.TTDIFFXS36_R21_allhad_resolved.root'
    dataset_bfixed = 'user.rpoggi.410471.PhPy8EG.DAOD_TOPQ1.e6337_e5984_s3126_r9364_r9315_p3629.TTDIFFXS34_R21_allhad_resolved.root'

    #no_bfixed = get_base_events(list((args.xsttbar_dir / dataset_nobfixed).glob('*.root')))
    bfixed = get_base_events(list((args.xsttbar_dir / dataset_bfixed).glob('*.root')))


    fig = plt.figure()
    bins_chi = np.linspace(0, 40, 400)
    #plt.hist(no_bfixed['reco_Chi2Fitted'], bins=bins_chi, histtype='step',
    #         label='$\chi^2$ no bfixed')
    plt.hist(bfixed['reco_Chi2Fitted'], bins=bins_chi, histtype='step',
             label='$\chi^2$ bfixed')
    plt.legend()
    plt.grid()
    plt.savefig('chi2_dist.png')

    #no_bfixed = no_bfixed[no_bfixed['reco_Chi2Fitted'] < 10]
    bfixed = bfixed[bfixed['reco_Chi2Fitted'] < 10]


    tj_top1 = np.concatenate(list(tj_distr(args.npz)))

    fig = plt.figure()
    bins_m = np.linspace(0, 500, 100)
    #plt.hist(no_bfixed['reco_t1_m'] / 1000, bins=bins_m, histtype='step',
    #         label='$\chi^2 < 10$ no bfixed', density=True)
    plt.hist(bfixed['reco_t1_m'] / 1000, bins=bins_m, histtype='step',
             label='$\chi^2 < 10$ bfixed', density=True)
    plt.hist(tj_top1, bins=bins_m, histtype='step', label='Top 1 (pred)',
             density=True)
    plt.xlabel('$t_1$ mass [GeV]')
    plt.legend()
    plt.grid()
    plt.savefig('mass_dist.png')

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
