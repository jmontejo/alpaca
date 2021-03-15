import ROOT

ntype='hydra'

by_jet = False
list_max_jet = [6,7,8,9,10,11,12]

by_mass = False
list_masses=[900,1400,2400]

by_file = True
files = [
    '/eos/user/c/crizzi/RPV/alpaca/results/alpaca_8j_hydra_3layers_UDS_1400_onlyBaseline/outputtree_test.root',
    '/eos/user/c/crizzi/RPV/alpaca/results/alpaca_8j_hydra_3layers_UDS_1400_isSig/outputtree_test.root',
    '/eos/user/c/crizzi/RPV/alpaca/results/alpaca_8j_hydra_3layers_UDS_1400_isSig_isGluon/outputtree_test.root'
]

def print_eff(tree, jet_sel,max_jet):
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

        has_truth_wrttot = round(100*(has_truth)/total,1)
        alpaca_good_mt1_wrttot = round(100*(n_alpaca_good_mt1)/total,1)
        alpaca_good_mt2_wrttot = round(100*(n_alpaca_good_mt2)/total,1)
        alpaca_good_both_wrttot = round(100*(n_alpaca_good_both)/total,1)
        
        #print(j, ' & ', chi2_good_mt1, '\\% & ', chi2_good_mt2, '\\% & ', chi2_good_both, '\\% & ', alpaca_good_mt1, '\\% & ', alpaca_good_mt2, '\\% & ', alpaca_good_both,'\\% \\\\')
        #print(max_jet,' & ',j.replace('njets>=5','Inclusive'), ' & ', alpaca_good_mt1, '\\% & ', alpaca_good_mt2, '\\% & ', alpaca_good_both,
        print(max_jet,' & ', has_truth_wrttot , '\\% & ', alpaca_good_mt1, '\\% & ', alpaca_good_mt2, '\\% & ', alpaca_good_both,
              '\\% & ', alpaca_good_mt1_wrttot, '\\% & ', alpaca_good_mt2_wrttot, '\\% & ', alpaca_good_both_wrttot,
        '\\% \\\\')
        print('\\hline')

if by_jet:
    for max_jet in list_max_jet:
        #max_jet=10
        infile_name='/eos/user/c/crizzi/RPV/alpaca/results/alpaca_'+str(max_jet)+'j_'+ntype+'_3layers_UDS_1400/outputtree_test.root'
        infile = ROOT.TFile.Open(infile_name,"READ")
        tree = infile.Get("tree")    
        #jet_sel = ['njets=='+str(ij) for ij in range(6,max_jet)] + ['njets>='+str(max_jet),'njets>=5']
        jet_sel = ['njets>=5']    
        print_eff(tree, jet_sel, max_jet)

if by_mass:
    for m in list_masses:
        if m==1400:
            infile_name='/eos/user/c/crizzi/RPV/alpaca/results/alpaca_8j_'+ntype+'_3layers_UDS_1400/outputtree_test.root'
        else:
            infile_name='/eos/user/c/crizzi/RPV/alpaca/results/alpaca_8j_'+ntype+'_3layers_UDS_1400/outputtree_test_mg'+str(m)+'.root'
        infile = ROOT.TFile.Open(infile_name,"READ")
        tree = infile.Get("tree")    
        #jet_sel = ['njets=='+str(ij) for ij in range(6,max_jet)] + ['njets>='+str(max_jet),'njets>=5']
        jet_sel = ['njets>=5']    
        print_eff(tree, jet_sel,m)

if by_file:
    for ifile,infile_name in enumerate(files):
        infile = ROOT.TFile.Open(infile_name,"READ")
        tree = infile.Get("tree")    
        #jet_sel = ['njets=='+str(ij) for ij in range(6,max_jet)] + ['njets>='+str(max_jet),'njets>=5']
        jet_sel = ['njets>=5']    
        print_eff(tree, jet_sel,ifile)
