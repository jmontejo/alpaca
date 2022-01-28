import logging
import os

import torch
torch.set_num_threads(1)
from alpaca.batch import BatchManager
from progressbar import progressbar

import pandas as pd

import alpaca.log
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from math import sqrt, floor

from torch.utils.data import DataLoader
import numpy as np
from alpaca.plot import get_roc_auc

__all__ = ['BaseMain']


log = logging.getLogger(__name__)


class BaseMain:

    def __init__(self, args):
        self.args = args
        from itertools import accumulate
        self.boundaries = list(accumulate([0]+args.outputs))
        self.losses = {cat:[] for cat in ['total']+args.categories}


    def get_output_dir(self):
        return self.args.output_dir / self.args.tag

    def get_model(self, args):
        if args.simple_nn:
            from alpaca.nn.simple import SimpleNN
            return SimpleNN((self.args.jets+self.args.extras)*(4+self.args.nextrafields) + self.args.nscalars, self.args.totaloutputs, fflayers=args.fflayers)
        elif args.hydra:
            from alpaca.nn.hydra import Hydra
            log.info('  FeedForwardHead intermediate layers')            
            return Hydra(self.args.jets+self.args.extras, args.ncombos, fflayers=args.fflayers,nscalars=self.args.nscalars, nextrafields=self.args.nextrafields)
        else: #ColaLola is default
            if args.per_jet:
                from alpaca.nn.colalolaperjet import CoLaLoLaPerJet
                return CoLaLoLaPerJet(self.args.jets,self.args.extras, args.ncombos, self.args.totaloutputs, nscalars=self.args.nscalars,nextrafields=self.args.nextrafields,fflayers=args.fflayers)
            else:
                from alpaca.nn.colalola import CoLaLoLa
                return CoLaLoLa(self.args.jets+self.args.extras, args.ncombos, self.args.totaloutputs, nscalars=self.args.nscalars,nextrafields=self.args.nextrafields,fflayers=args.fflayers)

    def run(self):
        args = self.args
        test_sample = args.test_sample if args.test_sample >= 0 else self.bm.get_nr_events()
        output_dir = self.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        param_file = output_dir / 'NN.pt'

        alpaca.log.setup_logger(file_path=output_dir / 'alpaca.log')
        log.debug('Alpaca has been started and can finally log')
        log.debug(self.args)

        log.info('Nr. of events: %s', self.bm.get_nr_events())

        if args.train:

            model = self.get_model(args)
            opt = torch.optim.Adam(model.parameters())
            self.bm.is_consistent(args)
            log.debug('BatchManager contents is consistent')

            nr_train = floor(sqrt(self.bm.get_nr_events()-test_sample))
            if args.fast: nr_train = int(sqrt(nr_train))
            batch_size = nr_train
            log.info('Training: %s iterations - batch size %s', nr_train, batch_size)
            for i in progressbar(range(nr_train)):
                model.train()
                opt.zero_grad()
                
                train_torch_batch = self.bm.get_torch_batch(batch_size, start_index=i * batch_size + test_sample)
                X, Y = train_torch_batch[0], train_torch_batch[1]  
                P = model(X)
                Y = Y.reshape(-1, args.totaloutputs)
                
                loss = {'total':0}                
                for i,cat in enumerate(args.categories):
                    Pi = P[:,self.boundaries[i] : self.boundaries[i+1]]
                    Yi = Y[:,self.boundaries[i] : self.boundaries[i+1]]                    
                    loss[cat] = torch.nn.functional.binary_cross_entropy(Pi, Yi)
                    # give more weight to signal events
                    #weight = Yi*9 + 1
                    #loss[cat] = torch.nn.functional.binary_cross_entropy(Pi, Yi, weight=weight)
                    loss['total'] += loss[cat]

                for key, val in loss.items():
                    self.losses[key].append(float(val))
                loss["total"].backward()
                opt.step()
            log.debug('Finished training')
            torch.save(model, param_file )

            fig = plt.figure()
            for losstype, lossvals in self.losses.items():
                plt.plot(lossvals, label=losstype)
            plt.ylim([0,1.5])
            plt.legend(ncol=2)
            plt.figtext(0.1,0.9,self.args.tag)
            plt.savefig(str(output_dir / 'losses.png'))
            plt.close()

        else:
            if os.path.exists(param_file):
                model = torch.load(param_file)
            else:
                log.error("Running without training but the model file {} is not present".format(param_file))
        model.eval()
        #def plots(self): ## should store the NN and then do the plotting as a separate step
        output_dir = self.get_output_dir()
        # Run for performance
        print("test sample:" , test_sample)
        test_torch_batch = self.bm.get_torch_batch(test_sample)
        X,Y = test_torch_batch[0], test_torch_batch[1]
        _X = X.data.numpy()
        _Y = Y.data.numpy()
        # if len(test_torch_batch) > 2: spec = test_torch_batch[2]            
        print('Evaluating on validation saample')
        _P_list=[]
        for batch in DataLoader(X, batch_size=250):
            P_appo  = model(batch)
            _P_appo = P_appo.data.numpy()
            _P_list.append(_P_appo)
        #FIXME, think about many bm plots
        _P = np.vstack(_P_list)
        flatdict = {}

        if self.args.nscalars:
            _X = _X[:,:-self.args.nscalars]
        _X = _X.reshape(X.shape[0],self.args.jets+self.args.extras,4+self.args.nextrafields)        

        # Write results to file with analysis-specific function
        if args.write_output:
            self.write_output(test_torch_batch, _P)

        if not args.no_truth: # Only for samples for which I have truth inf
            for i,(cat,jets) in enumerate(zip(args.categories, args.outputs)):
                Pi = _P[:,self.boundaries[i] : self.boundaries[i+1]]
                Yi = _Y[:,self.boundaries[i] : self.boundaries[i+1]]

                for ijet in range(jets):

                    log.info('Plot ROC curve, category %s and output num %d'%(cat,ijet))
                    fpr, tpr, thr = roc_curve(Yi[:,ijet], Pi[:,ijet])
                    roc_auc = auc(fpr, tpr)
                    
                    fig = plt.figure()
                    plt.plot(fpr, tpr, color='darkorange',
                             label='ROC curve (area = {:.2f})'.format(roc_auc))
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                #plt.title('Receiver operating characteristic')
                    plt.legend(loc="lower right")
                    plt.savefig(str(output_dir / 'roc_curve{}_{}_{}.png'.format((len(args.label_roc)>0)*'_'+args.label_roc,cat,ijet)))
                    plt.close()

    def write_output(self, torch_batch, P):
        ''' Writes the result into a file 
            Takes as input the return value of get_torch_batch and P = model(X)
        '''
        raise NotImplementedError('Please implement write_output in your BatchManager','See write_output in batch.py for documentation')
