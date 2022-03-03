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
        self.losses = {cat:[] for cat in ['total','validation']+args.categories}


    def get_output_dir(self):
        return self.args.output_dir / self.args.tag

    def get_model(self, args):
        do_multi_class = False 

        if hasattr(args,'multi_class'):
            if args.multi_class > 1:
                do_multi_class = True 

        if args.simple_nn:
            from alpaca.nn.simple import SimpleNN
            return SimpleNN((self.args.jets+self.args.extras)*(4+self.args.nextrafields) + self.args.nscalars, self.args.totaloutputs, fflayers=args.fflayersm, do_multi_class = do_multi_class)

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
        test_sample = args.test_sample if args.test_sample >= 0 else len(self.bm)//10
        output_dir = self.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        param_file = output_dir / 'NN.pt'

        alpaca.log.setup_logger(file_path=output_dir / 'alpaca.log')
        log.debug('Alpaca has been started and can finally log')
        log.debug(self.args)

        log.info('Nr. of events: %s', len(self.bm))
        self.bm.is_consistent(args)
        log.debug('BatchManager contents is consistent')

        bmtest, bmtrain = torch.utils.data.random_split(self.bm, [test_sample, len(self.bm)-test_sample])


        if args.train:

            model = self.get_model(args)
            opt = torch.optim.Adam(model.parameters())

            if args.fast:
                indices = np.random.choice(len(bmtrain), len(bmtrain)//10, replace=False)
                bmtrain = torch.utils.data.Subset(bmtrain, indices)
                log.debug('Using fast, will keep only {} events'.format(len(bmtrain)))

            batch_size = floor(sqrt(len(bmtrain)))
            nr_train = batch_size
            log.info('Training: %s iterations - batch size %s', nr_train, batch_size)

            validX, validY = bmtest[:]
            validY = validY.reshape(-1, args.totallabels)
            loader = DataLoader(bmtrain, batch_size=batch_size)
            for i, (X,Y) in enumerate(progressbar(loader)):
                model.train()
                opt.zero_grad()
                

                P = model(X)
                Y = Y.reshape(-1, args.totallabels)
                
                loss = {'total':0}
                
                if args.multi_class > 1:
                    Y_mclass = Y.flatten().type(torch.LongTensor)
                    P_mclass = P.reshape(-1,args.multi_class)
                    criterion = torch.nn.CrossEntropyLoss()
                    loss['total'] = criterion(P_mclass, Y_mclass)

                else:
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

                #Validation loss
                if i%args.validation_steps==0:
                    model.eval()
                    validP = model(validX)
                    if args.multi_class > 1:
                        Y_mclass = validY.flatten().type(torch.LongTensor)
                        P_mclass = validP.reshape(-1,args.multi_class)
                        validloss = criterion(P_mclass, Y_mclass)
                    else:
                        validloss = torch.nn.functional.binary_cross_entropy(validP, validY)
                    self.losses['validation'].append(float(validloss))

            log.debug('Finished training')
            torch.save(model, param_file )

            fig = plt.figure()
            for losstype, lossvals in self.losses.items():
                if losstype == 'validation':
                    plt.plot([l*args.validation_steps for l in range(len(lossvals))], lossvals, label=losstype)
                else:
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

        with torch.no_grad():
            model.eval()
            #def plots(self): ## should store the NN and then do the plotting as a separate step
            output_dir = self.get_output_dir()
            # Run for performance
            _P_list=[]
            print('Evaluating on validation sample')

            batch_size = 250
            validloader = DataLoader(self.bm, batch_size=batch_size)
            for X,Y in progressbar(validloader):
                _X = X.data.numpy()
                _Y = Y.data.numpy()
                actualbatch_size = _X.shape[0]

                P_appo  = model(X)
                # normalize _P for multiclass classification
                if self.args.multi_class > 1:
                    n_class = self.args.multi_class
                    if self.args.first_jet_gluino:
                        P_appo = P_appo.reshape(actualbatch_size, self.args.jets-1, self.args.multi_class)
                    else:
                        P_appo = P_appo.reshape(actualbatch_size, self.args.jets, self.args.multi_class)
                    P_softmax = torch.nn.functional.softmax(P_appo, dim = 2)
                    P_softmax = np.swapaxes(P_softmax,1,2)
                    P_softmax = P_softmax.reshape(actualbatch_size, -1)
                    _P_appo = P_softmax.data.numpy()
                else:
                    _P_appo = P_appo.data.numpy()
                _P_list.append(_P_appo)
            #FIXME, think about many bm plots
            _P = np.vstack(_P_list)
            flatdict = {}


        X, Y = self.bm[:]
        _Y = Y.data.numpy()
        # Write results to file with analysis-specific function
        if args.write_output:
            self.write_output((X,Y), _P)
        if not args.no_truth: # Only for samples for which I have truth inf
            # print out a few events here
            # print(_Y[:5])
            # print(_P[:5])
            for i,(cat,jets) in enumerate(zip(args.categories, args.outputs)):
                Pi = _P[:,self.boundaries[i] : self.boundaries[i+1]]
                
                # redefine Yi
                if args.multi_class > 1:
                    n_class = args.multi_class
                    n_jet = args.jets
                    Yi = np.tile(_Y,n_class)
                    end = 0
                    for class_i in range(n_class):
                        start = end
                        end = (start+n_jet-1 if args.first_jet_gluino else start+n_jet)
                        Yi[:,start:end] = (_Y == class_i).astype('int')
                    Yi = Yi[:,self.boundaries[i] : self.boundaries[i+1]]
                else:
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
