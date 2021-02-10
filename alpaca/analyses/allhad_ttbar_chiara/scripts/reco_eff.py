import ROOT
max_jet=8
ntype='hydra'
infile_name='/eos/user/c/crizzi/RPV/alpaca/results/alpaca_'+str(max_jet)+'j_'+ntype+'_3layers_UDS_1400/outputtree_test.root'
infile = ROOT.TFile.Open(infile_name,"READ")
tree = infile.Get("tree")

jet_sel = ['njets=='+str(ij) for ij in range(6,max_jet)] + ['njets>='+str(max_jet),'njets>=5']

for j in jet_sel:
    #print(j)
    total = float(tree.GetEntries("1 &&"+j))
    has_truth = float(tree.GetEntries("has_truth &&"+j))    
    #chi2_good_mt1 = round(100*(tree.GetEntries("chi2_good_mt1==1 &&"+j))/has_truth,1)
    #chi2_good_mt2 = round(100*(tree.GetEntries("chi2_good_mt2==1 &&"+j))/has_truth,1)
    #chi2_good_both = round(100*(tree.GetEntries("chi2_good_mt2==1 && chi2_good_mt1==1 &&"+j))/has_truth,1)
    n_alpaca_good_mt1 = tree.GetEntries("alpaca_good_mt1==1 &&"+j)
    alpaca_good_mt1 = round(100*(n_alpaca_good_mt1)/has_truth,1)
    n_alpaca_good_mt2 = tree.GetEntries("alpaca_good_mt2==1 &&"+j)
    alpaca_good_mt2 = round(100*(n_alpaca_good_mt2)/has_truth,1)
    n_alpaca_good_both = tree.GetEntries("alpaca_good_mt2==1 && alpaca_good_mt1==1 &&"+j)
    alpaca_good_both = round(100*(n_alpaca_good_both)/has_truth,1)

    alpaca_good_mt1_wrttot = round(100*(n_alpaca_good_mt1)/total,1)
    alpaca_good_mt2_wrttot = round(100*(n_alpaca_good_mt2)/total,1)
    alpaca_good_both_wrttot = round(100*(n_alpaca_good_both)/total,1)

    #print(j, ' & ', chi2_good_mt1, '\\% & ', chi2_good_mt2, '\\% & ', chi2_good_both, '\\% & ', alpaca_good_mt1, '\\% & ', alpaca_good_mt2, '\\% & ', alpaca_good_both,'\\% \\\\')
    print(j.replace('njets>=5','Total'), ' & ', alpaca_good_mt1, '\\% & ', alpaca_good_mt2, '\\% & ', alpaca_good_both,
          ' & ', alpaca_good_mt1_wrttot, '\\% & ', alpaca_good_mt2_wrttot, '\\% & ', alpaca_good_both_wrttot,
          '\\% \\\\')
    print('\\hline')
