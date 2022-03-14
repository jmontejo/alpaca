import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import torch

__all__ = ['plot_cola_weights', 'plot_hydra_weights', 'plot_roc_curve', 'get_roc_auc',
           'plot_score_roc', 'plot_true_ISR', 'plot_index_ISR',
           'plot_topmatch']

def plot_confusion_matrix(Y, P,filename):
    categP = np.argmax(P, axis=2)
    Y = Y.astype(int)
    cm = confusion_matrix(Y.flatten(), categP.flatten())
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.savefig(filename)
    plt.close()

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

def get_roc_auc(pred,truth):
    fpr, tpr, thr = roc_curve(truth, pred)
    return auc(fpr, tpr)

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
    for sample in ["Test","Train","6-jet"]:
        diff_ttbar = np.array([p[t==1].min() - p[t==0].max() for t,p in zip(truth["ttbar"][sample],pred["ttbar"][sample])])
        plt.hist(diff_ttbar, label='Diff ({})'.format(sample), **histargs)
    plt.legend(loc="upper center" )
    plt.savefig(str(output_dir / 'diff_topmatch.png'))
    plt.close()


def reco_gluino_mass(X, P, firstJetGluino=False, deterministic=False, tempfortopk=100, g1only=False):

    # X = X.data.numpy()
    # P = P.data.numpy()
    n=2
    debug = False
    if debug:
        print(X.shape)
        print(P.shape)
        print(X[:n])
        print(P[:n])


    is_ISR_P = P[:,:,0]
    is_lead_P = P[:,:,1]
    is_sublead_P = P[:,:,2]

    emptyjets =  X[:,:,3]==0
    if deterministic:
        is_ISR_P[emptyjets] = 99 #in-place change prevents gradients, can not be used for smooth

    jet_4p = X
    n_event = jet_4p.shape[0]
    n_jet = 8
    n_ISR = n_jet-6
    n_gluino = 3
    
    # find the ISR jets
    # Exclude jets with highest ISR score until there are 6 gluino jets left

    v, i = torch.sort(is_ISR_P, dim=1, descending=True)
    ISR_threshold = v[:,1]
    ISR_mask = (is_ISR_P >= ISR_threshold[:,None]).int()
    ISR_mask_smooth = smooth_topk(is_ISR_P, 2, temp=tempfortopk)
    if debug:
        print(is_ISR_P.shape, ISR_mask.shape, ISR_mask.shape)
        print(ISR_mask[:n])

    #zero out ISR jets
    #renorm_lead_P = is_lead_P/(is_lead_P+is_sublead_P) #problematic with smooth approx
    renorm_lead_P = is_lead_P
    if deterministic:
        renorm_lead_P = renorm_lead_P*(1-ISR_mask)
    else:
        renorm_lead_P = renorm_lead_P*(1-ISR_mask_smooth)
    v, i = torch.sort(renorm_lead_P, dim=1, descending=True)
    lead_threshold = v[:,2]
    lead_mask = (renorm_lead_P >= lead_threshold[:,None]).int()
    sublead_mask = 1 - lead_mask - ISR_mask

    lead_mask_smooth = smooth_topk(renorm_lead_P, 3, ISR_mask_smooth, temp=tempfortopk)
    sublead_mask_smooth = 1 - lead_mask_smooth - ISR_mask_smooth

    if deterministic:
        jet_4p_lead = torch.mul(jet_4p,lead_mask[:,:,None])
        jet_4p_sublead = torch.mul(jet_4p,sublead_mask[:,:,None])
    else:   
        jet_4p_lead = torch.mul(jet_4p,lead_mask_smooth[:,:,None])
        jet_4p_sublead = torch.mul(jet_4p,sublead_mask_smooth[:,:,None])

    g1_4p = torch.sum(jet_4p_lead,axis=1)
    g2_4p = torch.sum(jet_4p_sublead,axis=1)

    # calculate the gluino mass squared
    g1_m2 = torch.square(g1_4p[:,3]) - torch.square(g1_4p[:,0]) - torch.square(g1_4p[:,1]) - torch.square(g1_4p[:,2])
    g2_m2 = torch.square(g2_4p[:,3]) - torch.square(g2_4p[:,0]) - torch.square(g2_4p[:,1]) - torch.square(g2_4p[:,2])
    
    if torch.amin(g1_m2)< 0 or (not g1only and torch.amin(g2_m2)< 0):
        print('Warning: negative mass squared exists.')
        #print(torch.sum(g1_m2< 0),torch.sum(g2_m2< 0))
        #examples = g2_m2< 0
        #print(is_ISR_P[examples])
        #print(is_lead_P[examples])
        #print(is_sublead_P[examples])
        #print(renorm_lead_P[examples])
        #print(ISR_mask_smooth[examples])
        #print(lead_mask_smooth[examples])
        #print(sublead_mask_smooth[examples])
        #print(jet_4p[examples])
        #sys.exit(2)
    
    # # create a boolean arry of whether there is a jet at each entry
    # lead_zero_jet = lead_triplet[:,:,3] <= 0
    # sublead_zero_jet = sublead_triplet[:,:,3] <= 0
    # no_zero_in_lead = torch.array_equal(lead_zero_jet, torch.zeros_like(lead_zero_jet))
    # no_zero_in_sublead = torch.array_equal(sublead_zero_jet, torch.zeros_like(sublead_zero_jet))
    # if not no_zero_in_lead or not no_zero_in_sublead:
    #     print('Error:')
    #     print('Do all the jets in the leading triplet exist? ', no_zero_in_lead)
    #     print('Do all the jets in the subleading triplet exist? ', no_zero_in_sublead)
    #     print('Number of zero jets in leading triplet: ', torch.sum(lead_zero_jet))
    #     print('Number of zero jets in subleading triplet: ', torch.sum(sublead_zero_jet))
    #     #print('from top score: ', jet_4p[torch.any(lead_zero_jet,axis=1)][:,:,3])
    #     #print('from top score: ', from_top_P[torch.any(lead_zero_jet,axis=1)])
    #     print('is lead score: ', gjet_is_lead_P[torch.any(lead_zero_jet,axis=1)])
    #     print('is sublead score: ', gjet_is_sublead_P[torch.any(lead_zero_jet,axis=1)])
    #     print('ISR score: ', is_ISR_P[torch.any(lead_zero_jet,axis=1)])
    #     print('E: ', jet_4p[torch.any(lead_zero_jet,axis=1),3])
    #     sys.exit(3)

    if g1only:
        return torch.abs(g1_m2),torch.abs(g1_m2)

    return torch.abs(g1_m2),torch.abs(g2_m2)

def smooth_topk(tensor, k, startmask=None, axis=1, temp=100, weightsquare=False, logistic=False, logitf=10, meaninsteadofmax=False, relu=True):
    ''' Usually topk returns the k highest elements in a tensor, which allows to build
        a mask to only combine those jets into a mass. However this is not differentiable
        so I pick a smooth approximation and combine all jets with almost 0-1 weights
        From https://stats.stackexchange.com/questions/444832/is-there-something-like-softmax-but-for-top-k-values 
        The value of temp=80 is optimized to make as strong as possible maxima without overflowing, assuming inputs are 0-1
    '''

    if startmask is not None:
        x = startmask
    else:
        x = torch.zeros_like(tensor)

    if meaninsteadofmax:
        offset = torch.amax(tensor, axis, keepdim=True)
    else:
        offset = torch.amax(tensor, axis, keepdim=True)

    exptensor = torch.exp(((tensor-offset)*temp).to(dtype=torch.float64))
    for _ in range(k):
        x = x + multisoftmax(exptensor,1-x, axis, temp, weightsquare, logistic, logitf)
        if relu:
            x = -torch.nn.functional.relu(-x + 1) + 1

    if startmask is not None:
        x = x - startmask
    return x

def multisoftmax(exptensor, w, axis, temp, weightsquare, logistic, logitf):
    a = exptensor*w
    weight = a/torch.sum(a, keepdim=True, axis=axis)
    if weightsquare:
        weight = weight*weight
    if logistic:
        weight = 1./(1. + torch.exp(-(weight-0.5)*logitf))

    return weight

