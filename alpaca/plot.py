import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

__all__ = ['plot_cola_weights', 'plot_hydra_weights', 'plot_roc_curve',
           'plot_score_roc', 'plot_true_ISR', 'plot_index_ISR',
           'plot_topmatch']


def plot_cola_weights(model, output_dir):
        fig = plt.figure()
        plt.imshow(model.cola.w_combo.data.numpy())
        plt.savefig(str(output_dir / 'w_combo.png'))
        plt.close()


def plot_hydra_weights(model, output_dir):
        fig = plt.figure()
        plt.imshow(model.isr_head.cola.w_combo.data.numpy())
        plt.savefig(str(output_dir / 'w_combo_isr.png'))
        plt.close()
        fig = plt.figure()
        plt.imshow(model.decay_head.cola.w_combo.data.numpy())
        plt.savefig(str(output_dir / 'w_combo_decay.png'))
        plt.close()


def plot_roc_curve(pred,truth,filename):
    fpr, tpr, thr = roc_curve(truth, pred)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()


def plot_score_roc(pred, truth, discriminant, output_dir, plotlabels, irange):
    for ijet in irange:
        fig = plt.figure()
        for sample in ["Test","Train"]:
            pred_jet = pred[discriminant][sample][:,ijet]
            truth_jet = truth[discriminant][sample][:,ijet]
            histargs = {"bins":50, "range":(0.,1.), "density":True, "histtype":'step'}
            if sample=="Test":
                histargs["alpha"] = 0.5
                histargs["fill"] = True
            plt.hist(pred_jet[truth_jet==1],
                     label='{} ({})'.format(plotlabels[0],sample), **histargs)
            plt.hist(pred_jet[truth_jet==0],
                     label='{} ({})'.format(plotlabels[1],sample), **histargs)
        plt.legend(loc="upper center")
        plt.savefig(str(output_dir / 'score_{}_jet{}.png'.format(discriminant,ijet)))
        plt.close()

        filen_roc = str(output_dir / 'roc_curve_{}_jet{}.png'.format(discriminant,ijet))
        plot_roc_curve(pred[discriminant]["Test"][:,ijet], truth[discriminant]["Test"][:,ijet], filen_roc)


def plot_true_ISR(pred, truth, output_dir):
    fig = plt.figure()
    histargs = {"bins":50, "range":(0.,1.), "density":True, "histtype":'step'}
    for sample in ["Test","Train","6-jet"]:
        max_ISRjet = np.array([p[t==0].max() for t,p in zip(truth["ISR"][sample],pred["ISR"][sample])])
        plt.hist(max_ISRjet, label=sample, **histargs)
    plt.legend()
    plt.savefig(str(output_dir / 'score_max_ISR.png'))
    plt.close()

    fig = plt.figure()
    for sample in ["Test","Train","6-jet"]:
        min_topjet = np.array([p[t==1].min() for t,p in zip(truth["ISR"][sample],pred["ISR"][sample])])
        plt.hist(min_topjet, label=sample, **histargs)
    plt.legend()
    plt.savefig(str(output_dir / 'score_min_topjet.png'))
    plt.close()

    fig = plt.figure()
    for sample in ["Test","Train","6-jet"]:
        diff_ISR = np.array([p[t==1].min() - p[t==0].max() for t,p in zip(truth["ISR"][sample],pred["ISR"][sample])])
        plt.hist(diff_ISR, label=sample, **histargs)
    plt.legend()
    plt.savefig(str(output_dir / 'score_diff_ISR.png'))
    plt.close()

def plot_index_ISR(pred, truth, output_dir):
    fig = plt.figure()
    histargs = {"bins":7, "range":(0,7), "density":True, "histtype":'step'}
    for sample in ["Test","Train","6-jet"]:
        plt.hist(pred["ISR"][sample].argmin(1), label='Pred ({})'.format(sample), **histargs)
        plt.hist(truth["ISR"][sample].argmin(1), label='True ({})'.format(sample),
                 alpha=0.5, fill=True, **histargs)
    plt.legend(loc="upper center")
    plt.savefig(str(output_dir / 'argmin.png'))
    plt.close()


def plot_topmatch(pred, truth, output_dir):
    histargs = {"bins":50, "range":(0.,1.), "density":True, "histtype":'step'}
    fig = plt.figure()
    for sample in ["Test","Train","6-jet"]:
        mask_top1 = truth["ttbar"][sample].astype(np.bool)
        pred_top1 = pred["ttbar"][sample][mask_top1]
        pred_top2 = pred["ttbar"][sample][~mask_top1]
        plt.hist(pred_top1, label='Top 1 ({})'.format(sample), **histargs)
        plt.hist(pred_top2, label='Top 2 ({})'.format(sample), alpha=0.5, fill=True, **histargs)
    plt.legend(loc="upper center" )
    plt.savefig(str(output_dir / 'topmatch.png'))
    plt.close()

    fig = plt.figure()
    histargs = {"bins":50, "range":(-1,1.), "density":True, "histtype":'step'}
    for sample in ["Test","Train"]: #,"6-jet"]: #6-jet is actuall not all partons, crashes
        diff_ttbar = np.array([p[t==1].min() - p[t==0].max() for t,p in zip(truth["ttbar"][sample],pred["ttbar"][sample])])
        plt.hist(diff_ttbar, label='Diff ({})'.format(sample), **histargs)
    plt.legend(loc="upper center" )
    plt.savefig(str(output_dir / 'diff_topmatch.png'))
    plt.close()
