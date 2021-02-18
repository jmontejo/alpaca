import logging
#import os

#from progressbar import progressbar

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from math import sqrt, floor
import sys
#import numpy as np

eos='/eos/user/c/crizzi/RPV/alpaca/results/'
csvname='NNoutput_test.csv'
models=['alpaca_sigbkg_12j_colalola_3layers_UDS_1400',
        'alpaca_sigbkg_12j_colalola_3layers_UDBUDS_1400',
        'alpaca_sigbkg_12j_colalola_3layers_UDBUDS_1400_weight10',
        'alpaca_sigbkg_12j_colalola_3layers_UDBUDS_1400_lessQCD']
labels = [l.replace('alpaca_sigbkg_12j_colalola_3layers_','').replace('_',' ') for l in models]
csv_list = [eos+'/'+m+'/'+csvname for m in models]
coly='tagged_true'
colp='tagged'
name_can = 'roc_comp_log'
colors=['darkorange','dodgerblue','hotpink','forestgreen', 'navy', 'gold']

fig = plt.figure()

for i,csv in enumerate(csv_list):
    logging.info('Looking at '+csv)
    df = pd.read_csv(csv)
    Y = df[coly].values
    P = df[colp].values
    logging.debug(Y.shape)
    logging.debug(P.shape)
    fpr, tpr, thr = roc_curve(Y, P)
    roc_auc = auc(fpr, tpr)
    plt.plot(tpr, 1./fpr, color=colors[i],
             label=labels[i]+' (ROC area = {:.2f})'.format(roc_auc))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
plt.ylabel('QCD Rejection')
plt.xlabel('Signal Efficiency')
#plt.xscale('log')
plt.yscale('log')
plt.grid(which='both', axis='both')
#plt.title('Receiver operating characteristic')
plt.legend(loc="upper right")
plt.savefig(name_can+'.png')
plt.close()
