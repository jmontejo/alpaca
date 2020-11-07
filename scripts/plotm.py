import pandas as pd
import numpy as np 
from root_pandas import to_root
import ROOT
from array import array
import uproot
import progressbar

filename_chi2 = '/Users/crizzi/lavoro/SUSY/RPV/alpaca_mio/alpaca_example_files/AT/test_in.410471.root'
tname_chi2 = 'nominal'
f_chi2 = uproot.open(filename_chi2)
t_chi2 = f_chi2[tname_chi2]
print(t_chi2.GetEntries())
df_chi2 = t_chi2.pandas.df(['eventNumber', 'reco_Chi2Fitted', 'reco_t1_m', 'reco_t2_m', 'jet_pt*'], flatten=False)
df_chi2['njets'] = df_chi2['jet_pt'].apply(lambda x: len(x))
df_chi2['njets_25'] = df_chi2['jet_pt'].apply(lambda x: len([j for j in x if j > 25000]))
df_chi2['pt_jet_0'] = df_chi2['jet_pt'].apply(lambda x: x[0]/1000.)
df_chi2['pt_jet_0_merge'] = round(df_chi2['pt_jet_0'],4)
df_chi2 = df_chi2[['eventNumber', 'reco_Chi2Fitted', 'reco_t1_m', 'reco_t2_m', 'njets', 'njets_25','pt_jet_0','pt_jet_0_merge']]
print(df_chi2.shape)
print(df_chi2.head(15))

filename_alpaca='/Users/crizzi/lavoro/SUSY/RPV/alpaca_mio/mytest.csv'
df_alpaca=pd.read_csv(filename_alpaca)
df_alpaca['pt_jet_0'] = np.sqrt(df_alpaca['jet_px_0']*df_alpaca['jet_px_0'] + df_alpaca['jet_py_0']*df_alpaca['jet_py_0'])/1000.
df_alpaca['pt_jet_0_merge'] = round(df_alpaca['pt_jet_0'], 4)
print(df_alpaca.shape)
print(df_alpaca[['pt_jet_0','pt_jet_0_merge']].head())


df = pd.merge(df_alpaca[['pt_jet_0_merge']], df_chi2, left_on='pt_jet_0_merge', right_on='pt_jet_0_merge', how='outer')
print(df.shape)
print(df.head())
#print(df.columns)
fname='df_alpaca.root'
tname='tree_alpaca'

to_root(df, fname, key=tname)
# branches
# 'jet_px_0', 'jet_py_0', 'jet_pz_0', 'jet_e_0' ... 'jet_e_6'
# 'from_top_0' ... 'from_top_6'
# 'same_as_lead_0' ... 'same_as_lead_4' 
# 'is_b_0' ... 'is_b_5'
# 'from_top_0_true' ... 'from_top_6_true'
# 'same_as_lead_0_true' ... 'same_as_lead_4_true' 
# 'is_b_0_true' ... 'is_b_5_true'

fname_out = 'outtree.root'
tname_out = 'tree_alpaca'
f_out = ROOT.TFile.Open(fname_out,'RECREATE')
t_out = ROOT.TTree(tname_out, tname_out)

mt1_reco  = array('d',[0])
mt1_true  = array('d',[0])
mt2_reco  = array('d',[0])
mt2_true  = array('d',[0])
t_out.Branch("mt1_reco",  mt1_reco, "mt1_reco/D")
t_out.Branch("mt2_reco",  mt2_reco, "mt2_reco/D")
t_out.Branch("mt1_true",  mt1_true, "mt1_true/D")
t_out.Branch("mt2_true",  mt2_true, "mt2_true/D")

f=ROOT.TFile.Open(fname,'READ')
t=f.Get(tname)
nentries=t.GetEntries()
for ientry in progressbar.progressbar(range(nentries)):
    t.GetEntry(ientry)
    jets_all = []
    jet_vars = [
        (t.jet_px_0, t.jet_py_0, t.jet_pz_0, t.jet_e_0),
        (t.jet_px_1, t.jet_py_1, t.jet_pz_1, t.jet_e_1),
        (t.jet_px_2, t.jet_py_2, t.jet_pz_2, t.jet_e_2),
        (t.jet_px_3, t.jet_py_3, t.jet_pz_3, t.jet_e_3),
        (t.jet_px_4, t.jet_py_4, t.jet_pz_4, t.jet_e_4),
        (t.jet_px_5, t.jet_py_5, t.jet_pz_5, t.jet_e_5),
        (t.jet_px_6, t.jet_py_6, t.jet_pz_6, t.jet_e_6)
        ]
    njets= len(jet_vars)
    for i in range(njets):
        jets_all.append(ROOT.TLorentzVector())
    for i,v in enumerate(jet_vars):
        jets_all[i].SetPxPyPzE(v[0]/1000., v[1]/1000., v[2]/1000., v[3]/1000.) # convert to GeV
    
    from_top = [t.from_top_0, t.from_top_1, t.from_top_2, t.from_top_3, t.from_top_4, t.from_top_5, t.from_top_6]
    val, idx_isr = min((val, idx_isr) for (idx_isr, val) in enumerate(from_top))
    # print(val, idx_isr)
    pred_from_top = [1 for i in range(njets)]
    pred_from_top[idx_isr]=0
    # print(pred_from_top)
    jets = [j for i,j in enumerate(jets_all) if pred_from_top[i]>0] # remove jet from ISR

    is_b = [t.is_b_0, t.is_b_1, t.is_b_2, t.is_b_3, t.is_b_4, t.is_b_5]
    # find first b
    val, idx_b1 = max((val, idx_b1) for (idx_b1, val) in enumerate(is_b))
    is_b_appo = is_b.copy()
    is_b_appo[idx_b1]=0 # put it temporary at zero to find second b
    val, idx_b2 = max((val, idx_b2) for (idx_b2, val) in enumerate(is_b_appo))
    pred_is_b = [0 for i in range(njets)]
    pred_is_b[idx_b1]=1
    pred_is_b[idx_b2]=1
    # print(pred_is_b)
    # print('Look at same_as_lead')
    same_as_lead = [t.same_as_lead_0, t.same_as_lead_1, t.same_as_lead_2, t.same_as_lead_3, t.same_as_lead_4]
    # print(same_as_lead)
    val, idx_sal1 = max((val, idx_sal1) for (idx_sal1, val) in enumerate(same_as_lead))
    same_as_lead_appo = same_as_lead.copy()
    same_as_lead_appo[idx_sal1]=0    
    val, idx_sal2 = max((val, idx_sal2) for (idx_sal2, val) in enumerate(same_as_lead_appo))
    # print(idx_sal1, idx_sal1)
    pred_same_as_lead = [0 for i in range(njets)]
    pred_same_as_lead[idx_sal1]=1
    pred_same_as_lead[idx_sal2]=1
    pred_same_as_lead.insert(0,1) 
    # print(pred_same_as_lead)
    t1_list = [t for i,t in enumerate(jets) if pred_same_as_lead[i]>0]
    # print(t1_list)
    t1 = t1_list[0]+t1_list[1]+t1_list[2]
    t2_list = [t for i,t in enumerate(jets) if pred_same_as_lead[i]<1]
    t2 = t2_list[0]+t2_list[1]+t2_list[2]

    # true info
    true_from_top = [t.from_top_0_true, t.from_top_1_true, t.from_top_2_true, t.from_top_3_true, t.from_top_4_true, t.from_top_5_true, t.from_top_6_true]
    jets_true = [j for i,j in enumerate(jets_all) if true_from_top[i]>0] # remove jet from ISR
    true_is_b = [t.is_b_0_true, t.is_b_1_true, t.is_b_2_true, t.is_b_3_true, t.is_b_4_true, t.is_b_5_true]
    true_same_as_lead = [t.same_as_lead_0_true, t.same_as_lead_1_true, t.same_as_lead_2_true, t.same_as_lead_3_true, t.same_as_lead_4_true]
    true_same_as_lead.insert(0,1)
    t1_list_true = [t for i,t in enumerate(jets_true) if true_same_as_lead[i]>0]
    t1_true = t1_list_true[0]+t1_list_true[1]+t1_list_true[2]
    t2_list_true = [t for i,t in enumerate(jets_true) if true_same_as_lead[i]<1]
    t2_true = t2_list_true[0]+t2_list_true[1]+t2_list_true[2]

    if t2.Pt() > t1.Pt(): t1,t2=t2,t1 # call t1 the top with leading pt
    if t2_true.Pt() > t1_true.Pt(): t1_true,t2_true=t2_true,t1_true  # call t1 the top with leading pt

    mt1_reco[0] = t1.M()
    mt2_reco[0] = t2.M()
    mt1_true[0] = t1_true.M()
    mt2_true[0] = t2_true.M()
    t_out.Fill()

    #if ientry%1000==0:
    #    print(ientry)
    #    print('M1 reco:',t1.M(),      ' M2 reco:',t2.M())
    #    print('M1 true:',t1_true.M(), ' M2 true:',t2_true.M())

f_out.Write()
f_out.Close()
f.Close()
