import numpy as np
from matplotlib import pyplot as plt

import sys
datadir = sys.argv[1]
filein = "{}/data.npz".format(datadir)
data = np.load(filein)

def minv(a):
    return np.sqrt(a[:,3]**2 - a[:,2]**2 - a[:,1]**2 - a[:,0]**2)

for sample in ["Test","Train","6-jet"]:
    jets = data["jets_{}".format(sample)]*1e-3 # Convert to GeV
    nevents = len(jets)

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
    #
    if jets.shape[1]>6:
        ISR_pred[np.arange(nevents), ISR_pred_score.argmin(1)] = 0
    #
    _ttbar_pred_score = np.array(ttbar_pred_score) # temp var to modify
    # select first (correct for extra index)
    ttbar_pred[np.arange(nevents), ttbar_pred_score.argmax(1)+1] = 1
    # zero out the max score
    _ttbar_pred_score[np.arange(nevents), ttbar_pred_score.argmax(1)] = 0
    # select second
    ttbar_pred[np.arange(nevents), _ttbar_pred_score.argmax(1)+1] = 1
    #
    jets_ttbar_pred = jets[ISR_pred.astype(np.bool)].reshape(nevents,6,4)
    jets_top1_pred = jets_ttbar_pred[ttbar_pred.astype(np.bool)].reshape(nevents,3,4)
    jets_top2_pred = jets_ttbar_pred[~ttbar_pred.astype(np.bool)].reshape(nevents,3,4)
    minv_top1_pred = minv(jets_top1_pred.sum(1))
    minv_top2_pred = minv(jets_top2_pred.sum(1))

    # for i in range(10):
    #     print('\n\n',i)
    #     print(jets[i])
    #     print(ISR_truth[i])
    #     print(ISR_pred[i])
    #     print(ttbar_truth[i])
    #     print(ttbar_pred[i])

    histargs = {"bins":50, "range":(0.,250.), "density":False, "histtype":'step'}

    fig = plt.figure()
    plt.hist(minv_top1_truth, label='Top 1 (truth)', **histargs, alpha=0.5, fill=True)
    plt.hist(minv_top2_truth, label='Top 2 (truth)', **histargs, alpha=0.5, fill=True)
    plt.hist(minv_top1_pred, label='Top 1 (pred)', **histargs)
    plt.hist(minv_top2_pred, label='Top 2 (pred)', **histargs)
    plt.legend(loc="upper left")
    plt.savefig('{}/top_mass_{}.png'.format(datadir,sample))
    plt.close()

    perfectISR = np.equal(ISR_pred,ISR_truth).sum(1)==jets.shape[1]
    print("Perfect ISR fraction ({}) = {} / {} = {:.3f}".format(sample,perfectISR.sum(),nevents,float(perfectISR.sum())/nevents))
    perfectmatch = perfectISR & (np.equal(ttbar_pred,ttbar_truth).sum(1)==6)
    print("Perfect match fraction ({}) = {} / {} = {:.3f}".format(sample,perfectmatch.sum(),nevents,float(perfectmatch.sum())/nevents))
