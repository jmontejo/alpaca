import pandas as pd
import numpy as np 
from root_pandas import to_root
import ROOT
from array import array
import uproot
import progressbar
import argparse


col_truth = ['eventNumber','from_top_0_true',
             'from_top_1_true', 'from_top_2_true', 'from_top_3_true',
             'from_top_4_true', 'from_top_5_true', 'from_top_6_true',
             'same_as_lead_0_true', 'same_as_lead_1_true', 'same_as_lead_2_true',
             'same_as_lead_3_true', 'same_as_lead_4_true', 'is_b_0_true',
             'is_b_1_true', 'is_b_2_true', 'is_b_3_true', 'is_b_4_true',
             'is_b_5_true']

def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-alpaca-truth',     default='NNoutput_test_truth.csv',        help="Input csv from alpaca with truth info")
    parser.add_argument('--input-alpaca-no-truth',     default='NNoutput_test_noTruth.csv',        help="Input csv from alpaca without truth info")
    parser.add_argument('--merged-df',     default='merged_df_test.csv',        help="Merged df name")
    parser.add_argument('--merged-df-root',     default='merged_df_test.root',        help="Merged df name when stored as TTree")
    parser.add_argument('--output-tree',     default='tree',        help="Merged df name")
    parser.add_argument('--build-df',    action='store_true' ,       help="Create the merged dataframe and store it as root file")
    parser.add_argument('--pt-order',    action='store_true' ,       help="Order tops by pT")
    parser.add_argument('--train-sample',    action='store_true' ,       help="Look at the group of root files corresponding to trainine sample")
    parser.add_argument('--no-input-root',    action='store_true' ,       help="Do not add the information in the original ROOT file")
    parser.add_argument('--no-truth',    action='store_true' ,       help="Do not add the information in the alpaca file with truth info")
    parser.add_argument('--output',     default='outtree_nopTchoice.root',        help="Name of output ROOT file")
    parser.add_argument('--weights',     default=[],  nargs='*',        help="Weights for output tree as in the csv file")
    return parser.parse_args()


def get_chi2_df(files_chi2, suff='', only_mass=False):
    df_chi2_list=[]
    print('Loading ROOT files')
    for f in  progressbar.progressbar(files_chi2):
        tname_chi2 = 'nominal'
        f_chi2 = uproot.open(f)
        t_chi2 = f_chi2[tname_chi2]
        df_chi2_list.append(t_chi2.pandas.df(['eventNumber', 'reco_Chi2Fitted', 'reco_t1_m', 'reco_t2_m', 'jet_pt*','reco_bjets','reco_DRbb','reco_DRbWMax'], flatten=False))

    df_chi2 = pd.concat(df_chi2_list)    
    
    df_chi2['njets'] = df_chi2['jet_pt'].apply(lambda x: len(x))
    df_chi2['njets_55'] = df_chi2['jet_pt'].apply(lambda x: len([j for j in x if j > 55000]))
    df_chi2['njets_25'] = df_chi2['jet_pt'].apply(lambda x: len([j for j in x if j > 25000]))
    # selections from all had ttbar analysis, except the mass selection
    df_chi2['pass_reco_chi2'] = np.where((df_chi2['reco_Chi2Fitted']<10), 1, 0)
    df_chi2['pass_anatop'] = np.where((df_chi2['njets_55']>5) & (df_chi2['reco_bjets']==2)  & (df_chi2['reco_DRbb']>2) & (df_chi2['reco_DRbWMax']<2.2), 1, 0)
    if only_mass:
        df_chi2 = df_chi2[['eventNumber', 'reco_Chi2Fitted', 'reco_t1_m', 'reco_t2_m', 'pass_reco_chi2']]
    else:
        df_chi2 = df_chi2[['eventNumber', 'reco_Chi2Fitted', 'reco_t1_m', 'reco_t2_m', 'njets', 'njets_55','njets_25','pass_reco_chi2','pass_anatop']]
    if len(suff)>0:
        col_name = [c+'_'+suff if 'reco' in c else c for c in df_chi2.columns]
        df_chi2.columns = col_name
    return df_chi2


def build_and_store_df(args, files_chi2_nobfixed, files_chi2_bfixed):
    df_to_concat=[]
    if not args.no_input_root:
        df_chi2_nobfixed = get_chi2_df(files_chi2_nobfixed,'nobfixed')
        df_chi2_nobfixed['has_chi2'] = 1
        df_chi2_bfixed = get_chi2_df(files_chi2_bfixed,'bfixed',only_mass=True)
        # df_chi2_bfixed['has_chi2'] = 1
        print('df_chi2_nobfixed')
        print(df_chi2_nobfixed.shape)
        print(df_chi2_nobfixed.head())
        print('df_chi2_bfixed')
        print(df_chi2_bfixed.shape)
        print(df_chi2_bfixed.head())


    if not args.no_truth:
        df_alpaca_truth=pd.read_csv(args.input_alpaca_truth)
        df_alpaca_truth = df_alpaca_truth[col_truth]
        df_alpaca_truth['has_truth'] = 1
        print('df_alpaca_truth')
        print(df_alpaca_truth.shape)
        print(df_alpaca_truth.head())

    df_alpaca_noTruth=pd.read_csv(args.input_alpaca_no_truth)
    columns_reco = [c for c in df_alpaca_noTruth.columns if 'true' not in c]
    df_alpaca_noTruth = df_alpaca_noTruth[columns_reco]
    print('df_alpaca_noTruth')
    print(df_alpaca_noTruth.shape)
    print(df_alpaca_noTruth.head())


    if not args.no_truth:
        df_alpaca = pd.merge(df_alpaca_truth, df_alpaca_noTruth, left_on='eventNumber', right_on='eventNumber', how='outer')    
        print('df_alpaca (truth + noTruth)')
        print(df_alpaca.shape)
        print(df_alpaca.head())
    else:
        df_alpaca = df_alpaca_noTruth
        df_alpaca_noTruth['has_truth'] = 0

    df_alpaca.fillna(0, inplace=True) # put the truth info at zero where missing
    '''
    print('set_index to event')
    df_alpaca.set_index('eventNumber')
    df_chi2_nobfixed.set_index('eventNumber')
    df_chi2_bfixed.set_index('eventNumber')
    '''
    print('df_alpaca (truth + noTruth) after fillna')
    print(df_alpaca.shape)
    print(df_alpaca.head())
    
    if not args.no_input_root:
        # df = pd.concat([df_alpaca, df_chi2_nobfixed, df_chi2_bfixed], join='inner', axis=1)
        df = pd.merge(df_alpaca, df_chi2_nobfixed, left_on='eventNumber', right_on='eventNumber', how='inner')
        print('df (alpaca + chi2_nobfixed)')
        print(df.shape)
        print('null elements:',df.isnull().sum().sum())
        print(df.head())
        df = pd.merge(df, df_chi2_bfixed, left_on='eventNumber', right_on='eventNumber', how='inner')
        print('df (alpaca + chi2_nobfixed + chi2_bfixed)')
        print(df.shape)
        print('null elements:',df.isnull().sum().sum())
        print(df.head())
        print(df[['eventNumber','jet_px_0','jet_py_1','from_top_4_true','reco_t1_m_bfixed','reco_t2_m_bfixed']].head())
        print(df.columns)
        df = df.drop_duplicates(subset=['eventNumber'])
        print('df after drop duplicates')
        print(df.shape)
        print('null elements:',df.isnull().sum().sum())
        print(df.head())
    else:
        df = df_alpaca
        df['has_chi2'] = 0
    #for c in df.columns:
    #    print('na in', c, df[c].isna().sum())            
    # df.to_csv(args.merged_df)
    return df
    

def build_tree(args):

    output_m1=[]
    output_m2=[]

    f_out = ROOT.TFile.Open(args.output,'RECREATE')
    t_out = ROOT.TTree(args.output_tree, args.output_tree)

    pt1_reco  = array('d',[0])
    pt2_reco  = array('d',[0])
    mt1_reco  = array('d',[0])
    mt2_reco  = array('d',[0])
    mt1_true  = array('d',[0])
    mt1_random  = array('d',[0])
    mt1_chi2_bfixed  = array('d',[0])
    mt1_chi2_nobfixed  = array('d',[0])
    mt2_random  = array('d',[0])
    mt2_true  = array('d',[0])
    mt2_chi2_bfixed  = array('d',[0])
    mt2_chi2_nobfixed  = array('d',[0])
    has_truth  = array('i',[0])
    pass_reco_chi2_nobfixed  = array('i',[0])
    pass_reco_chi2_bfixed  = array('i',[0])
    pass_anatop  = array('i',[0])
    njets = array('i',[0])
    njets_25 = array('i',[0])
    njets_55 = array('i',[0])
    wei  = array('d',[0])


    t_out.Branch("mt1_reco",  mt1_reco, "mt1_reco/D")
    t_out.Branch("mt2_reco",  mt2_reco, "mt2_reco/D")
    t_out.Branch("pt1_reco",  pt1_reco, "pt1_reco/D")
    t_out.Branch("pt2_reco",  pt2_reco, "pt2_reco/D")
    t_out.Branch("mt1_random",  mt1_random, "mt1_random/D")
    t_out.Branch("mt2_random",  mt2_random, "mt2_random/D")
    t_out.Branch("mt1_true",  mt1_true, "mt1_true/D")
    t_out.Branch("mt2_true",  mt2_true, "mt2_true/D")
    t_out.Branch("mt1_chi2_bfixed",  mt1_chi2_bfixed, "mt1_chi2_bfixed/D")
    t_out.Branch("mt2_chi2_bfixed",  mt2_chi2_bfixed, "mt2_chi2_bfixed/D")
    t_out.Branch("mt1_chi2_nobfixed",  mt1_chi2_nobfixed, "mt1_chi2_nobfixed/D")
    t_out.Branch("mt2_chi2_nobfixed",  mt2_chi2_nobfixed, "mt2_chi2_nobfixed/D")

    t_out.Branch("njets",  njets, "njets/I")
    t_out.Branch("njets_25",  njets_25, "njets_25/I")
    t_out.Branch("njets_55",  njets_55, "njets_55/I")

    t_out.Branch("has_truth",  has_truth, "has_truth/I")
    t_out.Branch("pass_chi2_nobfixed",  pass_reco_chi2_nobfixed, "pass_chi2_nobfixed/I")
    t_out.Branch("pass_chi2_bfixed",  pass_reco_chi2_bfixed, "pass_chi2_bfixed/I")
    t_out.Branch("pass_anatop",  pass_anatop, "pass_anatop/I")
    
    t_out.Branch("weight",  wei, "weight/D")

    f=ROOT.TFile.Open(args.merged_df_root,'READ')
    t=f.Get(args.output_tree)

    '''
    # branches
    # 'jet_px_0', 'jet_py_0', 'jet_pz_0', 'jet_e_0' ... 'jet_e_6'
    # 'from_top_0' ... 'from_top_6'
    # 'same_as_lead_0' ... 'same_as_lead_4' 
    # 'is_b_0' ... 'is_b_5'
    # 'from_top_0_true' ... 'from_top_6_true'
    # 'same_as_lead_0_true' ... 'same_as_lead_4_true' 
    # 'is_b_0_true' ... 'is_b_5_true'
    '''

    nentries=t.GetEntries()
    for ientry in progressbar.progressbar(range(nentries)):
    # for ientry in range(3):
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
        nj= len(jet_vars)
        for i in range(nj):
            jets_all.append(ROOT.TLorentzVector())
        for i,v in enumerate(jet_vars):
            jets_all[i].SetPxPyPzE(v[0]/1000., v[1]/1000., v[2]/1000., v[3]/1000.) # convert to GeV

        from_top = [t.from_top_0, t.from_top_1, t.from_top_2, t.from_top_3, t.from_top_4, t.from_top_5, t.from_top_6]    
        is_b = [t.is_b_0, t.is_b_1, t.is_b_2, t.is_b_3, t.is_b_4, t.is_b_5]
        same_as_lead = [t.same_as_lead_0, t.same_as_lead_1, t.same_as_lead_2, t.same_as_lead_3, t.same_as_lead_4]

        import random
        from_top_random = [random.uniform(0,1) for i in range(7)]
        is_b_random = [random.uniform(0,1) for i in range(6)]
        same_as_lead_random = [random.uniform(0,1) for i in range(5)]

        def form_tops(jets_all, from_top, same_as_lead,  is_b):
            val, idx_isr = min((val, idx_isr) for (idx_isr, val) in enumerate(from_top))
            # print(val, idx_isr)
            pred_from_top = [1 for i in range(nj)]
            pred_from_top[idx_isr]=0
            # print(pred_from_top)
            jets = [j for i,j in enumerate(jets_all) if pred_from_top[i]>0] # remove jet from ISR        
            # find first b
            val, idx_b1 = max((val, idx_b1) for (idx_b1, val) in enumerate(is_b))
            is_b_appo = is_b.copy()
            is_b_appo[idx_b1]=0 # put it temporary at zero to find second b
            val, idx_b2 = max((val, idx_b2) for (idx_b2, val) in enumerate(is_b_appo))
            pred_is_b = [0 for i in range(nj)]
            pred_is_b[idx_b1]=1
            pred_is_b[idx_b2]=1
            # print(pred_is_b)
            # print('Look at same_as_lead')
            # print(same_as_lead)
            val, idx_sal1 = max((val, idx_sal1) for (idx_sal1, val) in enumerate(same_as_lead))
            same_as_lead_appo = same_as_lead.copy()
            same_as_lead_appo[idx_sal1]=0    
            val, idx_sal2 = max((val, idx_sal2) for (idx_sal2, val) in enumerate(same_as_lead_appo))
            # print(idx_sal1, idx_sal1)
            pred_same_as_lead = [0 for i in range(nj)]
            pred_same_as_lead[idx_sal1]=1
            pred_same_as_lead[idx_sal2]=1
            pred_same_as_lead.insert(0,1) 
            # print(pred_same_as_lead)
            t1_list = [t for i,t in enumerate(jets) if pred_same_as_lead[i]>0]
            # print(t1_list)
            t1 = t1_list[0]+t1_list[1]+t1_list[2]
            t2_list = [t for i,t in enumerate(jets) if pred_same_as_lead[i]<1]
            t2 = t2_list[0]+t2_list[1]+t2_list[2]
            if args.pt_order:
                if t2.Pt() > t1.Pt(): t1,t2=t2,t1 # call t1 the top with leading pt        
            return t1, t2

        t1, t2 = form_tops(jets_all, from_top, same_as_lead,  is_b)
        mt1_reco[0] = t1.M()
        mt2_reco[0] = t2.M()
        pt1_reco[0] = t1.Pt()
        pt2_reco[0] = t2.Pt()

        t1_random, t2_random = form_tops(jets_all, from_top_random, same_as_lead_random,  is_b_random)
        mt1_random[0] = t1_random.M()
        mt2_random[0] = t2_random.M()


        if t.has_truth:
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
            # call t1 the top with leading pt
            if args.pt_order:
                if t2_true.Pt() > t1_true.Pt(): t1_true,t2_true=t2_true,t1_true          
            mt1_true[0] = t1_true.M()
            mt2_true[0] = t2_true.M()
        else:
            mt1_true[0] = -99
            mt2_true[0] = -99

        has_truth[0] = int(t.has_truth)
        if t.has_chi2:
            pass_reco_chi2_nobfixed[0] = int(t.pass_reco_chi2_nobfixed)
            pass_reco_chi2_bfixed[0] = int(t.pass_reco_chi2_bfixed)
            pass_anatop[0] = int(t.pass_anatop)
            mt1_chi2_nobfixed[0] = t.reco_t1_m_nobfixed / 1000.
            mt2_chi2_nobfixed[0] = t.reco_t2_m_nobfixed / 1000.
            mt1_chi2_bfixed[0] = t.reco_t1_m_bfixed / 1000.
            mt2_chi2_bfixed[0] = t.reco_t2_m_bfixed / 1000.
            njets[0] =  t.njets
            njets_25[0] =  t.njets_25
            njets_55[0] =  t.njets_55
        else:
            pass_reco_chi2_nobfixed[0] = -99
            pass_reco_chi2_bfixed[0] = -99
            pass_anatop[0] = -99
            mt1_chi2_nobfixed[0] = -99
            mt2_chi2_nobfixed[0] = -99
            mt1_chi2_bfixed[0] = -99
            mt2_chi2_bfixed[0] = -99
            njets[0] =  -99
            njets_25[0] =  -99
            njets_55[0] =  -99

        wei[0]=1
        for w in args.weights: 
            readw = getattr(t, w)
            wei[0] = wei[0]*readw

        output_m1.append(mt1_reco[0])
        output_m2.append(mt2_reco[0])

        t_out.Fill()
        
        #if ientry%1000==0:
        #    print(ientry)
        #    print('M1 reco:',t1.M(),      ' M2 reco:',t2.M())
        #    print('M1 true:',t1_true.M(), ' M2 true:',t2_true.M())
        
    f_out.Write()
    f_out.Close()
    f.Close()
    return output_m1, output_m2
    

def main():
    args = options()

    folder_chi2_nobfixed = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/alpaca_tutorial/user.rpoggi.410471.PhPy8EG.DAOD_TOPQ1.e6337_e5984_s3126_r9364_r9315_p3629.TTDIFFXS36_R21_allhad_resolved.root/'
    files_chi2_nobfixed = ["user.rpoggi.18378247._000001.allhad_resolved.root",
                           "user.rpoggi.18378247._000002.allhad_resolved.root", 
                           "user.rpoggi.18378247._000003.allhad_resolved.root", 
                           "user.rpoggi.18378247._000004.allhad_resolved.root", 
                           "user.rpoggi.18378247._000005.allhad_resolved.root", 
                           "user.rpoggi.18378247._000006.allhad_resolved.root", 
                           "user.rpoggi.18378247._000007.allhad_resolved.root", 
                           "user.rpoggi.18378247._000008.allhad_resolved.root", 
                           "user.rpoggi.18378247._000009.allhad_resolved.root", 
                           "user.rpoggi.18378247._000001.allhad_resolved.root", 
                           "user.rpoggi.18378247._000010.allhad_resolved.root", 
                           "user.rpoggi.18378247._000011.allhad_resolved.root", 
                           "user.rpoggi.18378247._000012.allhad_resolved.root", 
                           "user.rpoggi.18378247._000013.allhad_resolved.root", 
                           "user.rpoggi.18378247._000014.allhad_resolved.root", 
                           "user.rpoggi.18378247._000015.allhad_resolved.root", 
                           "user.rpoggi.18378247._000016.allhad_resolved.root", 
                           "user.rpoggi.18378247._000017.allhad_resolved.root", 
                           "user.rpoggi.18378247._000018.allhad_resolved.root", 
                           "user.rpoggi.18378247._000019.allhad_resolved.root"]
    if args.train_sample:
        files_chi2_nobfixed = ["user.rpoggi.18378247._000020.allhad_resolved.root", 
                               "user.rpoggi.18378247._000021.allhad_resolved.root", 
                               "user.rpoggi.18378247._000022.allhad_resolved.root", 
                               "user.rpoggi.18378247._000023.allhad_resolved.root", 
                               "user.rpoggi.18378247._000024.allhad_resolved.root", 
                               "user.rpoggi.18378247._000025.allhad_resolved.root", 
                               "user.rpoggi.18378247._000026.allhad_resolved.root", 
                               "user.rpoggi.18378247._000027.allhad_resolved.root", 
                               "user.rpoggi.18378247._000028.allhad_resolved.root", 
                               "user.rpoggi.18378247._000029.allhad_resolved.root", 
                               "user.rpoggi.18378247._000030.allhad_resolved.root", 
                               "user.rpoggi.18378247._000031.allhad_resolved.root", 
                               "user.rpoggi.18378247._000032.allhad_resolved.root", 
                               "user.rpoggi.18378247._000033.allhad_resolved.root", 
                               "user.rpoggi.18378247._000034.allhad_resolved.root", 
                               "user.rpoggi.18378247._000035.allhad_resolved.root", 
                               "user.rpoggi.18378247._000036.allhad_resolved.root", 
                               "user.rpoggi.18378247._000037.allhad_resolved.root", 
                               "user.rpoggi.18378247._000038.allhad_resolved.root", 
                               "user.rpoggi.18378247._000039.allhad_resolved.root", 
                               "user.rpoggi.18378247._000040.allhad_resolved.root", 
                               "user.rpoggi.18378247._000041.allhad_resolved.root", 
                               "user.rpoggi.18378247._000042.allhad_resolved.root", 
                               "user.rpoggi.18378247._000043.allhad_resolved.root", 
                               "user.rpoggi.18378247._000044.allhad_resolved.root", 
                               "user.rpoggi.18378247._000045.allhad_resolved.root"]
        
    files_chi2_nobfixed = [folder_chi2_nobfixed + f for f in files_chi2_nobfixed] # add folder
    
    
    folder_chi2_bfixed = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/alpaca_tutorial/user.rpoggi.410471.PhPy8EG.DAOD_TOPQ1.e6337_e5984_s3126_r9364_r9315_p3629.TTDIFFXS34_R21_allhad_resolved.root/'
    files_chi2_bfixed = [ 'user.rpoggi.17773514._000001.allhad_resolved.root', 
                          'user.rpoggi.17773514._000002.allhad_resolved.root', 
                          'user.rpoggi.17773514._000003.allhad_resolved.root', 
                          'user.rpoggi.17773514._000004.allhad_resolved.root', 
                          'user.rpoggi.17773514._000005.allhad_resolved.root', 
                          'user.rpoggi.17773514._000006.allhad_resolved.root', 
                          'user.rpoggi.17773514._000007.allhad_resolved.root', 
                          'user.rpoggi.17773514._000008.allhad_resolved.root', 
                          'user.rpoggi.17773514._000009.allhad_resolved.root', 
                          'user.rpoggi.17773514._000010.allhad_resolved.root', 
                          'user.rpoggi.17773514._000011.allhad_resolved.root', 
                          'user.rpoggi.17773514._000012.allhad_resolved.root', 
                          'user.rpoggi.17773514._000013.allhad_resolved.root', 
                          'user.rpoggi.17773514._000014.allhad_resolved.root', 
                          'user.rpoggi.17773514._000015.allhad_resolved.root', 
                          'user.rpoggi.17773514._000016.allhad_resolved.root', 
                          'user.rpoggi.17773514._000017.allhad_resolved.root', 
                          'user.rpoggi.17773514._000018.allhad_resolved.root', 
                          'user.rpoggi.17773514._000019.allhad_resolved.root',
                          'user.rpoggi.17773514._000020.allhad_resolved.root',
                          'user.rpoggi.17773514._000021.allhad_resolved.root',
                          'user.rpoggi.17773514._000022.allhad_resolved.root',
                          'user.rpoggi.17773514._000023.allhad_resolved.root',
                          'user.rpoggi.17773514._000024.allhad_resolved.root',
                          'user.rpoggi.17773514._000025.allhad_resolved.root',
                          'user.rpoggi.17773514._000026.allhad_resolved.root',
                          'user.rpoggi.17773514._000027.allhad_resolved.root',
                          'user.rpoggi.17773514._000028.allhad_resolved.root',
                          'user.rpoggi.17773514._000029.allhad_resolved.root',
                          'user.rpoggi.17773514._000030.allhad_resolved.root',
                          'user.rpoggi.17773514._000031.allhad_resolved.root',
                          'user.rpoggi.17773514._000032.allhad_resolved.root',
                          'user.rpoggi.17773514._000033.allhad_resolved.root',
                          'user.rpoggi.17773514._000034.allhad_resolved.root',
                          'user.rpoggi.17773514._000035.allhad_resolved.root',
                          'user.rpoggi.17773514._000036.allhad_resolved.root',
                          'user.rpoggi.17773514._000037.allhad_resolved.root',
                          'user.rpoggi.17773514._000038.allhad_resolved.root',
                          'user.rpoggi.17773514._000039.allhad_resolved.root',
                          'user.rpoggi.17773514._000040.allhad_resolved.root',
                          'user.rpoggi.17773514._000041.allhad_resolved.root',
                          'user.rpoggi.17773514._000042.allhad_resolved.root']

    files_chi2_bfixed = [folder_chi2_bfixed + f for f in files_chi2_bfixed] # add folder

    if args.build_df:
        df = build_and_store_df(args, files_chi2_nobfixed, files_chi2_bfixed)
        to_root(df, args.merged_df_root, key=args.output_tree)

    m1, m2 = build_tree(args)
    df['m1_chiara'] = m1
    df['m2_chiara'] =  m2
    df.to_csv((args.output).replace('root','csv'))

if __name__=='__main__':
    main()

