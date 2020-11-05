import pandas as pd
import numpy as np 
from root_pandas import to_root
import ROOT

filename='/Users/crizzi/lavoro/SUSY/RPV/alpaca_mio/mytest.csv'
df=pd.read_csv(filename)
#print(df.head())
#print(df.columns)
fname='out.root'
tname='mytree'
to_root(df, fname, key=tname)

# branches
# 'jet_px_0', 'jet_py_0', 'jet_pz_0', 'jet_e_0' ... 'jet_e_6'
# 'from_top_0' ... 'from_top_6'
# 'same_as_lead_0' ... 'same_as_lead_4' 
# 'is_b_0' ... 'is_b_5'
# 'from_top_0_true' ... 'from_top_6_true'
# 'same_as_lead_0_true' ... 'same_as_lead_4_true' 
# 'is_b_0_true' ... 'is_b_5_true'

f=ROOT.TFile.Open(fname,'READ')
t=f.Get(tname)
nentries=t.GetEntries()
for ientry in range(nentries):
    t.GetEntry(ientry)
    jets = []
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
        jets.append(ROOT.TLorentzVector())
    for i,v in enumerate(jet_vars):
        jets[i].SetPxPyPzE(v[0]/1000., v[1]/1000., v[2]/1000., v[3]/1000.) # convert to GeV
    # print('mass of each jet')
    # for j in jets:
    #    print(j.M())
    from_top = [t.from_top_0, t.from_top_1, t.from_top_2, t.from_top_3, t.from_top_4, t.from_top_5, t.from_top_6]
    # print(from_top)
    val, idx_isr = min((val, idx_isr) for (idx_isr, val) in enumerate(from_top))
    # print(val, idx_isr)
    pred_from_top = [1 for i in range(njets)]
    pred_from_top[idx_isr]=0
    # print(pred_from_top)
    jets = [j for i,j in enumerate(jets) if pred_from_top[i]>0] # remove jet from ISR
    # print('mass of each jet')
    # for j in jets:
        # print(j.M())    
    is_b = [t.is_b_0, t.is_b_1, t.is_b_2, t.is_b_3, t.is_b_4, t.is_b_5]
    # print(is_b)
    val, idx_b1 = max((val, idx_b1) for (idx_b1, val) in enumerate(is_b))
    is_b_appo = is_b.copy()
    is_b_appo[idx_b1]=0    
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
    if ientry%1000==0:
        print(ientry)
        print('M1:',t1.M(), ' M2:',t2.M())
