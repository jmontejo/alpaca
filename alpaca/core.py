import logging

import torch
from alpaca.batch import BatchManager
from progressbar import progressbar

import alpaca.log
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from math import sqrt, floor


__all__ = ['BaseMain']


log = logging.getLogger(__name__)


class BaseMain:

    def __init__(self, args):
        self.args = args
        self.losses = []
        self.test_sample = 2000


    def get_output_dir(self):
        return self.args.output_dir / self.args.tag

    def get_model(self, args):
        from alpaca.nn.colalola import CoLaLoLa
        return CoLaLoLa(self.args.jets, 30, self.args.totaloutputs, fflayers=[200])

    def run(self):
        args = self.args
        output_dir = self.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        alpaca.log.setup_logger(file_path=output_dir / 'alpaca.log')
        log.debug('Alpaca has been started and can finally log')
        log.debug(self.args)

        log.info('Nr. of events: %s', self.train_bm.get_nr_events())

        model = self.get_model(args)

        opt = torch.optim.Adam(model.parameters())

        self.train_bm.is_consistent(args)
        log.debug('BatchManager contents is consistent')

        nr_train = floor(sqrt(self.train_bm.get_nr_events()-self.test_sample))
        nr_train = min(200,nr_train)
        batch_size = nr_train
        log.info('Training: %s iterations - batch size %s', nr_train, batch_size)
        for i in progressbar(range(nr_train)):
            model.train()
            opt.zero_grad()

            X, Y = self.train_bm.get_torch_batch(batch_size, start_index=i * batch_size + self.test_sample)
            P = model(X)
            Y = Y.reshape(-1, args.totaloutputs)

            loss = torch.nn.functional.binary_cross_entropy(P, Y)
            self.losses.append(float(loss))
            loss.backward()
            opt.step()
        log.debug('Finished training')

    #def plots(self): ## should store the NN and then do the plotting as a separate step
        output_dir = self.get_output_dir()

        fig = plt.figure()
        plt.plot(self.losses)
        plt.savefig(str(output_dir / 'losses.png'))

        # Run for performance
        X,Y = self.train_bm.get_torch_batch(self.test_sample)
        P  = model(X)
        _P = P.data.numpy()
        _Y = Y.data.numpy()


        log.info('Plot ROC curve')
        fpr, tpr, thr = roc_curve(_Y, _P)
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
        plt.savefig(str(output_dir / 'roc_curve.png'))
