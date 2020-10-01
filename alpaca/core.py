import logging

import torch
from alpaca.batch import BatchManager
from progressbar import progressbar

import alpaca.log

__all__ = ['BaseMain']


log = logging.getLogger(__name__)


class BaseMain:

    def __init__(self, args):
        self.args = args

    def get_output_dir(self):
        return self.args.output_dir / self.args.tag

    def get_model(self):
        from alpaca.nn.colalola import CoLaLoLa
        return CoLaLoLa(7, 30, 7 + 5 + 6, fflayers=[200])

    def run(self):
        args = self.args
        output_dir = self.get_output_dir()

        output_dir.mkdir(parents=True, exist_ok=True)

        alpaca.log.setup_logger(file_path=output_dir / 'alpaca.log')
        log.debug('Alpaca has been started and can finally log')
        log.debug(self.args)

        njets = 7
        bm = BatchManager(
            input_paths=args.input_files,
            input_categories=args.input_categories,
        )
        log.info('Nr. of events: %s', bm.get_nr_events())

        model = self.get_model()

        opt = torch.optim.Adam(model.parameters())
        losses = []

        nr_train = 250
        batch_size = 250
        log.info('Training: %s iterations - batch size %s', nr_train, batch_size)
        for i in progressbar(range(nr_train)):
            model.train()
            opt.zero_grad()

            X, Y = bm.get_torch_batch(batch_size, start_index=i * batch_size + 5000)
            P = model(X)
            Y = Y.reshape(-1, noutputs)

            loss = torch.nn.functional.binary_cross_entropy(P, Y)
            losses.append(float(loss))
            loss.backward()
            opt.step()

        log.debug('Finished training')

        fig = plt.figure()
        plt.plot(losses)
        plt.savefig(str(output_dir / 'losses.png'))

        # Run for performance
        X,Y = bm.get_torch_batch(5000, nr_train * batch_size)
        P, masses_sig = model(X)
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
