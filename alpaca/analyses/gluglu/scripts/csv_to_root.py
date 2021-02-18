import pandas as pd
import numpy as np 
from root_pandas import to_root
import ROOT
from array import array
import uproot
import progressbar
import argparse
import itertools

pd.set_option('display.max_columns', None)


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
    parser.add_argument('--jets', '-j',    default=7,        help="How many jets have been used to train alpaca? (atm max 10)", type=int)
    parser.add_argument('--weights',     default=[],  nargs='*',        help="Weights for output tree as in the csv file")
    parser.add_argument('--presel',   action='store_true',        help="Apply preselection")
    return parser.parse_args()

files_FT=['/eos/user/c/crizzi/RPV/ntuples/FT_merged_skim/qcd.root']
#files_FT=['/eos/user/c/crizzi/RPV/ntuples/FT_signal_020321_merged/mc16e/signal/504539.root']
def get_FT_df(files_FT, presel=False):
    df_FT_list=[]
    print('Loading ROOT files')
    for f in  progressbar.progressbar(files_FT):
        tname_FT = 'trees_SRRPV_'
        f_FT = uproot.open(f)
        t_FT = f_FT[tname_FT]
        df_FT_list.append(t_FT.pandas.df(['mcChannelNumber','mcEventWeight','eventNumber','pass_HLT_ht1000_L1J100','normweight','jet_pt'], flatten=False))

    df_FT = pd.concat(df_FT_list)    
    
    df_FT['njets'] = df_FT['jet_pt'].apply(lambda x: len(x))
    df_FT['njets_50'] = df_FT['jet_pt'].apply(lambda x: len([j for j in x if j > 50]))
    df_FT['njets_25'] = df_FT['jet_pt'].apply(lambda x: len([j for j in x if j > 25]))
    df_FT['ht'] = df_FT['jet_pt'].apply(lambda x: sum(x))
    df_FT['ht_50'] = df_FT['jet_pt'].apply(lambda x: sum([j for j in x if j > 50]))
    df_FT['ht_8j'] = df_FT['jet_pt'].apply(lambda x: sum(x[:8]))
    df_FT['jet_pt_0'] = df_FT['jet_pt'].apply(lambda x: x[0])
    df_FT['jet_pt_5'] = df_FT['jet_pt'].apply(lambda x: x[5] if len(x)>5 else 0)
    df_FT['jet_0_round'] = df_FT['jet_pt_0'].round(2)

    if presel:
        df_FT = df_FT[(df_FT.pass_HLT_ht1000_L1J100>0) & (df_FT.jet_pt_0>350) & (df_FT.ht_50>1100) & (df_FT.jet_pt_5>50) & ((df_FT.jet_pt_5/df_FT.jet_pt_0)>0.01)]

    cols_to_keep=[c for c in df_FT.columns if not 'jet_pt' in c]
    df_FT=df_FT[cols_to_keep]

    return df_FT


def build_and_store_df(args, files_FT):
    df_to_concat=[]
    if not args.no_input_root:
        df_FT = get_FT_df(files_FT, args.presel)
        df_FT['has_FT'] = 1
        print('df_FT')
        print(df_FT.shape)

    col_truth = ['event_number','from_top_0_true',
                 'from_top_1_true', 'from_top_2_true', 'from_top_3_true',
                 'from_top_4_true', 'from_top_5_true',
                 'same_as_lead_0_true', 'same_as_lead_1_true', 'same_as_lead_2_true',
                 'same_as_lead_3_true', 'same_as_lead_4_true', 'is_b_0_true',
                 'is_b_1_true', 'is_b_2_true', 'is_b_3_true', 'is_b_4_true',
                 'is_b_5_true']
    for ij in range(6,20):        
        if args.jets > ij:
            col_truth.append('from_top_'+str(ij)+'_true')

    if not args.no_truth:
        df_alpaca_truth=pd.read_csv(args.input_alpaca_truth)
        df_alpaca_truth = df_alpaca_truth[col_truth]
        df_alpaca_truth['has_truth'] = 1
        print('df_alpaca_truth')
        print(df_alpaca_truth.shape)
        #print(df_alpaca_truth.head())

    df_alpaca_noTruth=pd.read_csv(args.input_alpaca_no_truth)
    columns_reco = [c for c in df_alpaca_noTruth.columns if 'true' not in c]
    df_alpaca_noTruth = df_alpaca_noTruth[columns_reco]
    df_alpaca_noTruth['jet_0_round'] = np.sqrt( df_alpaca_noTruth['jet_px_0']**2 + df_alpaca_noTruth['jet_py_0']**2  ).round(2)
    print('df_alpaca_noTruth')
    print(df_alpaca_noTruth.shape)
    print(df_alpaca_noTruth.columns)
    #print(df_alpaca_noTruth.head())


    if not args.no_truth:
        df_alpaca = pd.merge(df_alpaca_truth, df_alpaca_noTruth, left_on='event_number', right_on='event_number', how='outer')    
        #df_alpaca = pd.merge(df_alpaca_truth, df_alpaca_noTruth, left_on='event_number', right_on='event_number', how='inner')    
        print('df_alpaca (truth + noTruth)')
        print(df_alpaca.shape)
        #print(df_alpaca.head())
    else:
        df_alpaca = df_alpaca_noTruth
        df_alpaca_noTruth['has_truth'] = 0

    df_alpaca.fillna(0, inplace=True) # put the truth info at zero where missing

    print('df_alpaca (truth + noTruth) after fillna')
    print(df_alpaca.shape)
    #print(df_alpaca.describe())
    #print(df_alpaca.head())
    
    if not args.no_input_root:        
        #df = pd.merge(df_alpaca, df_FT, left_on=['event_number','n_jets'], right_on=['eventNumber','njets'], how='inner')
        df = pd.merge(df_alpaca, df_FT, left_on=['event_number','jet_0_round'], right_on=['eventNumber','jet_0_round'], how='inner')
        #df = pd.merge(df_alpaca, df_FT, left_on=['event_number'], right_on=['eventNumber'], how='inner')
        print('df (alpaca + FT)')
        print(df.shape)
        print('null elements:',df.isnull().sum().sum())
        #print(df.head())
        #print(df.columns)
        df = df.drop_duplicates(subset=['event_number','jet_0_round'])
        print('df after drop duplicates')
        print(df.shape)
        print('null elements:',df.isnull().sum().sum())
        #print(df.head())
    else:
        df = df_alpaca
        df['has_FT'] = 0
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
    ht_8j = array('d',[0])
    ht = array('d',[0])
    ht_50 = array('d',[0])
    jet_pt_0 = array('d',[0])
    jet_pt_1 = array('d',[0])
    jet_pt_2 = array('d',[0])
    jet_pt_3 = array('d',[0])
    jet_pt_4 = array('d',[0])
    jet_pt_5 = array('d',[0])
    jet_pt_6 = array('d',[0])
    jet_pt_7 = array('d',[0])
    njets = array('i',[0])
    njets_25 = array('i',[0])
    njets_50 = array('i',[0])

    alpaca_good_mt1 = array('i',[0])
    alpaca_good_mt2 = array('i',[0])
    chi2_good_mt1 = array('i',[0])
    chi2_good_mt2 = array('i',[0])

    score_is_from_tt_1 = array('d',[0])
    score_is_from_tt_2 = array('d',[0])
    score_same_as_lead_1 = array('d',[0])
    score_same_as_lead_2 = array('d',[0])
    score_sum = array('d',[0])

    wei  = array('d',[0])

    #m_32_t1, m_32_t2, D_32_t1, D_32_t2, m_63_ijk, D2 # 3, 3, 1, 1, 2, 1
    m_32_t1 = array('d',[0,0,0])
    m_32_t2 = array('d',[0,0,0])
    D_32_t1 = array('d',[0])
    D_32_t2 = array('d',[0])
    m_63_ijk = array('d',[0,0])
    D2 = array('d',[0])
    # dphi_t1_t2, dR_t1_t2, dphi_t1, dR_t1, dphi_t2, dR_t2 # sizes: 1, 1, 2, 2, 2, 2
    dphi_t1_t2 = array('d',[0])
    dR_t1_t2 = array('d',[0])
    dphi_t1_min = array('d',[0])
    dphi_t1_max = array('d',[0])
    dR_t1_min = array('d',[0])
    dR_t1_max = array('d',[0])
    dphi_t2_min = array('d',[0])
    dphi_t2_max = array('d',[0])
    dR_t2_min = array('d',[0])
    dR_t2_max = array('d',[0])

    t_out.Branch("mt1_reco",  mt1_reco, "mt1_reco/D")
    t_out.Branch("mt2_reco",  mt2_reco, "mt2_reco/D")
    t_out.Branch("pt1_reco",  pt1_reco, "pt1_reco/D")
    t_out.Branch("pt2_reco",  pt2_reco, "pt2_reco/D")
    t_out.Branch("mt1_random",  mt1_random, "mt1_random/D")
    t_out.Branch("mt2_random",  mt2_random, "mt2_random/D")
    t_out.Branch("mt1_true",  mt1_true, "mt1_true/D")
    t_out.Branch("mt2_true",  mt2_true, "mt2_true/D")

    t_out.Branch("ht_8j",  ht_8j, "ht_8j/D")
    t_out.Branch("ht",  ht, "ht/D")
    t_out.Branch("ht_50",  ht_50, "ht_50/D")
    t_out.Branch("jet_pt_0",  jet_pt_0, "jet_pt_0/D")
    t_out.Branch("jet_pt_1",  jet_pt_1, "jet_pt_1/D")
    t_out.Branch("jet_pt_2",  jet_pt_2, "jet_pt_2/D")
    t_out.Branch("jet_pt_3",  jet_pt_3, "jet_pt_3/D")
    t_out.Branch("jet_pt_4",  jet_pt_4, "jet_pt_4/D")
    t_out.Branch("jet_pt_5",  jet_pt_5, "jet_pt_5/D")
    t_out.Branch("jet_pt_6",  jet_pt_6, "jet_pt_6/D")
    t_out.Branch("jet_pt_7",  jet_pt_7, "jet_pt_7/D")

    t_out.Branch("njets",  njets, "njets/I")
    t_out.Branch("njets_25",  njets_25, "njets_25/I")
    t_out.Branch("njets_50",  njets_50, "njets_50/I")

    t_out.Branch("has_truth",  has_truth, "has_truth/I")    
    t_out.Branch("alpaca_good_mt1",  alpaca_good_mt1, "alpaca_good_mt1/I")
    t_out.Branch("alpaca_good_mt2",  alpaca_good_mt2, "alpaca_good_mt2/I")

    t_out.Branch("score_is_from_tt_1",  score_is_from_tt_1, "score_is_from_tt_1/D")
    t_out.Branch("score_is_from_tt_2",  score_is_from_tt_2, "score_is_from_tt_2/D")
    t_out.Branch("score_same_as_lead_1",  score_same_as_lead_1, "score_same_as_lead_1/D")
    t_out.Branch("score_same_as_lead_2",  score_same_as_lead_2, "score_same_as_lead_2/D")
    t_out.Branch("score_sum",  score_sum, "score_sum/D")

    # dalitz variables
    t_out.Branch("m_32_t1",  m_32_t1, "m_32_t1[3]/D")
    t_out.Branch("m_32_t2",  m_32_t2, "m_32_t2[3]/D")
    t_out.Branch("D_32_t1",  D_32_t1, "D_32_t1/D")
    t_out.Branch("D_32_t2",  D_32_t2, "D_32_t2/D")
    t_out.Branch("m_63_ijk",  m_63_ijk, "m_63_ijk[2]/D")
    t_out.Branch("D2",  D2, "D2/D")

    # angular variables
    t_out.Branch("dphi_t1_t2", dphi_t1_t2, "dphi_t1_t2/D")
    t_out.Branch("dR_t1_t2", dR_t1_t2 , "dR_t1_t2/D")
    t_out.Branch("dphi_t1_min", dphi_t1_min , "dphi_t1_min/D")
    t_out.Branch("dphi_t1_max", dphi_t1_max , "dphi_t1_max/D")
    t_out.Branch("dR_t1_min", dR_t1_min , "dR_t1_min/D")
    t_out.Branch("dR_t1_max", dR_t1_max , "dR_t1_max/D")
    t_out.Branch("dphi_t2_min", dphi_t2_min , "dphi_t2_min/D")
    t_out.Branch("dphi_t2_max", dphi_t2_max , "dphi_t2_max/D")
    t_out.Branch("dR_t2_min", dR_t2_min , "dR_t2_min/D")
    t_out.Branch("dR_t2_max", dR_t2_max , "dR_t2_max/D")

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
    #nentries=1000 # chiara: tmp test
    for ientry in progressbar.progressbar(range(nentries)):
    # for ientry in range(5):
        t.GetEntry(ientry)
        jets_all = []
        jet_vars = [
            (t.jet_px_0, t.jet_py_0, t.jet_pz_0, t.jet_e_0),
            (t.jet_px_1, t.jet_py_1, t.jet_pz_1, t.jet_e_1),
            (t.jet_px_2, t.jet_py_2, t.jet_pz_2, t.jet_e_2),
            (t.jet_px_3, t.jet_py_3, t.jet_pz_3, t.jet_e_3),
            (t.jet_px_4, t.jet_py_4, t.jet_pz_4, t.jet_e_4),
            (t.jet_px_5, t.jet_py_5, t.jet_pz_5, t.jet_e_5)
        ]
        if args.jets > 6:
            jet_vars.append((t.jet_px_6, t.jet_py_6, t.jet_pz_6, t.jet_e_6))
        if args.jets > 7:
            jet_vars.append((t.jet_px_7, t.jet_py_7, t.jet_pz_7, t.jet_e_7))
        if args.jets > 8:
            jet_vars.append((t.jet_px_8, t.jet_py_8, t.jet_pz_8, t.jet_e_8))
        if args.jets > 9:
            jet_vars.append((t.jet_px_9, t.jet_py_9, t.jet_pz_9, t.jet_e_9))
        if args.jets > 10:
            jet_vars.append((t.jet_px_10, t.jet_py_10, t.jet_pz_10, t.jet_e_10))
        if args.jets > 11:
            jet_vars.append((t.jet_px_11, t.jet_py_11, t.jet_pz_11, t.jet_e_11))

        nj= len(jet_vars)
        if not nj == args.jets: 
            print('ERROR! nj == args.jets')

        for i in range(nj):
            jets_all.append(ROOT.TLorentzVector())
        for i,v in enumerate(jet_vars):
            jets_all[i].SetPxPyPzE(v[0], v[1], v[2], v[3]) # already in GeV

        from_top = [t.from_top_0, t.from_top_1, t.from_top_2, t.from_top_3, t.from_top_4, t.from_top_5]    
        if args.jets > 6:
            from_top.append( t.from_top_6)
        if args.jets > 7:
            from_top.append( t.from_top_7)
        if args.jets > 8:
            from_top.append( t.from_top_8)
        if args.jets > 9:
            from_top.append( t.from_top_9)
        if args.jets > 10:
            from_top.append( t.from_top_10)
        if args.jets > 11:
            from_top.append( t.from_top_11)

        is_b = [t.is_b_0, t.is_b_1, t.is_b_2, t.is_b_3, t.is_b_4, t.is_b_5]
        same_as_lead = [t.same_as_lead_0, t.same_as_lead_1, t.same_as_lead_2, t.same_as_lead_3, t.same_as_lead_4]

        import random
        from_top_random = [random.uniform(0,1) for i in range(nj)]
        is_b_random = [random.uniform(0,1) for i in range(6)]
        same_as_lead_random = [random.uniform(0,1) for i in range(5)]


        def d_32_from_m32(m_32):
            # D^2(3,2) = sum( (m(3,2)_{i,j} - 1/sqrt(3))^2  )
            D_32 = m_32 - (1/np.sqrt(3))
            D_32 = np.power(D_32, 2)
            D_32 = np.sum(D_32)
            return np.sqrt(D_32)
            
        def from_tlist_to_m32(t_list):
            t = t_list[0] + t_list[1] + t_list[2]
            den_t = t_list[0].M()**2 + t_list[1].M()**2 + t_list[2].M()**2 + t.M()**2
            # print('t.M()',t.M(),'  den_t',den_t)
            m_32 = []
            pairs = list(itertools.combinations(range(3), 2))
            for p in pairs:
                qi = t_list[p[0]]
                qj = t_list[p[1]]
                mij2 = (qi+qj).M() ** 2
                if den_t>0: m_32.append(mij2/den_t)
                else: m_32.append(0)
            # sqrt since now we have the square
            return np.sqrt(m_32)

        def form_m_63_ijk(t1_list, t2_list):
            t1 = t1_list[0] + t1_list[1] + t1_list[2]
            t2 = t2_list[0] + t2_list[1] + t2_list[2]
            t1t2 = t1 + t2
            den = 4*(t1t2.M()**2) + 6*np.sum([q.M()**2 for q in t1_list+t2_list])
            m_63_2 = [(t1.M()**2)/den, (t2.M()**2)/den]
            return np.sqrt(m_63_2)
            
        def form_d2(m_63_ijk, D_32_t1, D_32_t2):
            d2_a = (np.sqrt(m_63_ijk[0]**2 + D_32_t1**2) - (1./np.sqrt(20)))**2
            d2_b = (np.sqrt(m_63_ijk[1]**2 + D_32_t2**2) - (1./np.sqrt(20)))**2
            return d2_a + d2_b 

        def form_dalitz(t1_list,t2_list):
            # find min and max dR in first gluino
            m_32_t1 = from_tlist_to_m32(t1_list) # list of 3
            m_32_t2 = from_tlist_to_m32(t2_list) # list of 3
            D_32_t1 = d_32_from_m32(m_32_t1) # single value
            D_32_t2 = d_32_from_m32(m_32_t2) # single value
            m_63_ijk = form_m_63_ijk(t1_list, t2_list) # list of 2
            D2 = form_d2(m_63_ijk, D_32_t1, D_32_t2) # single value
            #print('m_32_t1:', m_32_t1)
            #print('m_32_t2:', m_32_t2)
            #print('D_32_t1:', D_32_t1)
            #print('D_32_t2:', D_32_t2)
            #print('')
            return m_32_t1, m_32_t2, D_32_t1, D_32_t2, m_63_ijk, D2 # 3, 3, 1, 1, 2, 1

        def min_max_angular_in_t(t_list, deltaR=False):
            pairs = list(itertools.combinations(range(3), 2))
            min_d = 9999
            max_d = 0 
            for p in pairs:
                qi = t_list[p[0]]
                qj = t_list[p[1]]
                if deltaR: d = qi.DeltaR(qj)
                else: d = qi.DeltaPhi(qj)
                if d > max_d: max_d = d
                if d < min_d: min_d = d
            return (min_d, max_d)

        def form_angular_var(t1_list,t2_list):
            t1 = t1_list[0] + t1_list[1] + t1_list[2]
            t2 = t2_list[0] + t2_list[1] + t2_list[2]
            dphi_t1_t2 = t1.DeltaPhi(t2)
            dR_t1_t2 = t1.DeltaR(t2)
            dphi_t1 = min_max_angular_in_t(t1_list, deltaR=False)
            dR_t1 = min_max_angular_in_t(t1_list, deltaR=True)
            dphi_t2 = min_max_angular_in_t(t2_list, deltaR=False)
            dR_t2 = min_max_angular_in_t(t2_list, deltaR=True)
            return dphi_t1_t2, dR_t1_t2, dphi_t1, dR_t1, dphi_t2, dR_t2 # sizes: 1, 1, 2, 2, 2, 2

        def form_tops(jets_all, from_top, same_as_lead,  is_b, njets):
            #print('Number of jets:', len(jets_all))
            #print("\n\n\n")
            #print('len jets_all:',len(jets_all))            
            idx_isr_list = []
            from_top_appo = from_top
            #print('from_top')
            #print(from_top)
            if not max(from_top)>0: return  0, 0, (0, 0, 0, 0, 0)
            for iisr in range(njets-6):                
                val, idx_isr = min((val, idx_isr) for (idx_isr, val) in enumerate(from_top_appo))
                from_top_appo[idx_isr] = 999 # for the next iteration I don't want this to be the minimum
                idx_isr_list.append(idx_isr)
                # print('  min and pos:',val, idx_isr)
            #print('len ISR list:', len(idx_isr_list))
            pred_from_top = [1 for i in range(nj)]
            for idx_isr in idx_isr_list:
                pred_from_top[idx_isr]=0
            #print('pred_from_top')
            #print(pred_from_top)

            jets = [j for i,j in enumerate(jets_all) if pred_from_top[i]>0] # remove jet from ISR        
            from_top_6j = [from_top[i] for i,j in enumerate(jets_all) if pred_from_top[i]>0] # keep track of ISR score of the 6 chosen jets
            # print('Number of jets considered:', len(jets))
            # find first b
            val, idx_b1 = max((val, idx_b1) for (idx_b1, val) in enumerate(is_b))
            is_b_appo = is_b.copy()
            is_b_appo[idx_b1]=0 # put it temporary at zero to find second b
            val, idx_b2 = max((val, idx_b2) for (idx_b2, val) in enumerate(is_b_appo))
            pred_is_b = [0 for i in range(6)]
            pred_is_b[idx_b1]=1
            pred_is_b[idx_b2]=1

            # print(pred_is_b)
            # print('Look at same_as_lead')
            # print("\n\n\n")
            # print("same_as_lead")
            # print(same_as_lead)
            val, idx_sal1 = max((val, idx_sal1) for (idx_sal1, val) in enumerate(same_as_lead))
            same_as_lead_appo = same_as_lead.copy()
            same_as_lead_appo[idx_sal1]=0    
            val, idx_sal2 = max((val, idx_sal2) for (idx_sal2, val) in enumerate(same_as_lead_appo))
            # print("idx_sal1, idx_sal2")
            # print(idx_sal1, idx_sal2)
            pred_same_as_lead = [0 for i in range(5)]
            pred_same_as_lead[idx_sal1]=1
            pred_same_as_lead[idx_sal2]=1
            pred_same_as_lead.insert(0,1) 
            # print("pred_same_as_lead")
            # print(pred_same_as_lead)
            t1_list = [t for i,t in enumerate(jets) if pred_same_as_lead[i]>0]
            is_from_top_1 = 0
            is_from_top_2 = 0
            sal_score_1 = 1 # len(pred_same_as_lead)=5
            sal_score_2 = 0
            for i,p in enumerate(pred_same_as_lead):
                if p>0:
                    is_from_top_1 += from_top_6j[i]
                    if i>0:
                        sal_score_1 += same_as_lead[i-1]
                if p<1: # should not need to check i>0 since pred_same_as_lead[0]=1 by construction
                    is_from_top_2 += from_top_6j[i]
                    sal_score_2 += (1.0 - same_as_lead[i-1])
            is_from_top_1 = is_from_top_1/3.0
            is_from_top_2 = is_from_top_2/3.0
            sal_score_1 = sal_score_1/3.0
            sal_score_2 = sal_score_2/3.0
            sum_score = (is_from_top_1 + is_from_top_2 + sal_score_1 + sal_score_2)/4.0
            #print("t1_list")
            #print(len(t1_list))
            t1 = t1_list[0]+t1_list[1]+t1_list[2]
            t2_list = [t for i,t in enumerate(jets) if pred_same_as_lead[i]<1]
            #print("t2_list")
            # print(len(t2_list))
            t2 = t2_list[0]+t2_list[1]+t2_list[2]
            if args.pt_order:
                if t2.Pt() > t1.Pt(): 
                    t1,t2=t2,t1 # call t1 the top with leading pt        
                    t1_list,t2_list = t2_list,t1_list # change lists as well
            # dalitz variables 
            dalitz_var = form_dalitz(t1_list,t2_list) # lenght of variables: 3, 3, 1, 1, 2, 1
            angular = form_angular_var(t1_list,t2_list)
            return t1, t2, (is_from_top_1, is_from_top_2, sal_score_1, sal_score_2, sum_score), dalitz_var, angluar

        t1, t2, scores, dalitz, angular = form_tops(jets_all, from_top, same_as_lead,  is_b, args.jets)
        mt1_reco[0] = t1.M()
        mt2_reco[0] = t2.M()
        pt1_reco[0] = t1.Pt()
        pt2_reco[0] = t2.Pt()
        
        score_is_from_tt_1[0] = scores[0]
        score_is_from_tt_2[0] = scores[1]
        score_same_as_lead_1[0] = scores[2]
        score_same_as_lead_2[0] = scores[3]
        score_sum[0] = scores[4]

        #m_32_t1, m_32_t2, D_32_t1, D_32_t2, m_63_ijk, D2
        m_32_t1[0] = dalitz[0][0]
        m_32_t1[1] = dalitz[0][1]
        m_32_t1[2] = dalitz[0][2]
        m_32_t2[0] = dalitz[1][0]
        m_32_t2[1] = dalitz[1][1]
        m_32_t2[2] = dalitz[1][2]
        D_32_t1[0] = dalitz[2]
        D_32_t2[0] = dalitz[3]
        m_63_ijk[0] = dalitz[4][0]
        m_63_ijk[1] = dalitz[4][1]
        D2[0] = dalitz[5]
        
        # dphi_t1_t2, dR_t1_t2, dphi_t1, dR_t1, dphi_t2, dR_t2 # sizes: 1, 1, 2, 2, 2, 2
        dphi_t1_t2[0] = angular[0]
        dR_t1_t2[0] = angular[1]
        dphi_t1_min[0] = angular[2][0]
        dphi_t1_max[0] = angular[2][1]
        dR_t1_min[0] = angular[3][0]
        dR_t1_max[0] = angular[3][1]
        dphi_t2_min[0] = angular[4][0]
        dphi_t2_max[0] = angular[4][1]
        dR_t2_min[0] = angular[5][0]
        dR_t2_max[0] = angular[5][1]

        t1_random, t2_random, scores_random, dalitz_random, angular_random = form_tops(jets_all, from_top_random, same_as_lead_random,  is_b_random, args.jets)
        mt1_random[0] = t1_random.M()
        mt2_random[0] = t2_random.M()

        njets_=0
        njets_25_=0
        njets_50_=0
        try:
            njets_ = int(getattr(t, 'n_jets'))
        except:
            for j in jets_all:
                if j.Pt()>0:
                    njets_ += 1
        #print('computing HT')
        ht_8j[0] = 0
        for ij,j in enumerate(jets_all):
            ht_8j[0] += j.Pt()
            #print('  adding:', j.Pt())
            if ij==0: jet_pt_0[0]=j.Pt()
            elif ij==1: jet_pt_1[0]=j.Pt()
            elif ij==2: jet_pt_2[0]=j.Pt()
            elif ij==3: jet_pt_3[0]=j.Pt()
            elif ij==4: jet_pt_4[0]=j.Pt()
            elif ij==5: jet_pt_5[0]=j.Pt()
            elif ij==6: jet_pt_6[0]=j.Pt()
            elif ij==7: jet_pt_7[0]=j.Pt()
            if j.Pt()>25:
                njets_25_ += 1
            if j.Pt()>50:
                njets_50_ += 1
        #print('ht:', ht_8j)

        if not args.no_input_root:
            ht[0] = t.ht
            ht_50[0] = t.ht_50
            njets_50[0] =  t.njets_50
            njets_25[0] =  t.njets_25
            njets[0] =  t.njets

        else:
            njets[0] = njets_
            njets_25[0] =  njets_25_
            njets_50[0] =  njets_50_
            ht = -999
            ht_50 = -999

        pass_reco_chi2_nobfixed[0] = -99
        pass_reco_chi2_bfixed[0] = -99
        pass_anatop[0] = -99
        mt1_chi2_nobfixed[0] = -99
        mt2_chi2_nobfixed[0] = -99
        mt1_chi2_bfixed[0] = -99
        mt2_chi2_bfixed[0] = -99
        
        if t.has_truth:
            # true info
            true_from_top = [t.from_top_0_true, t.from_top_1_true, t.from_top_2_true, t.from_top_3_true, t.from_top_4_true, t.from_top_5_true]
            if args.jets > 6:
                true_from_top.append(t.from_top_6_true)
            if args.jets > 7:
                true_from_top.append(t.from_top_7_true)
            if args.jets > 8:
                true_from_top.append(t.from_top_8_true)
            if args.jets > 9:
                true_from_top.append(t.from_top_9_true)
            if args.jets > 10:
                true_from_top.append(t.from_top_10_true)
            if args.jets > 11:
                true_from_top.append(t.from_top_11_true)
            jets_true = [j for i,j in enumerate(jets_all) if true_from_top[i]>0] # remove jet from ISR
            true_is_b = [t.is_b_0_true, t.is_b_1_true, t.is_b_2_true, t.is_b_3_true, t.is_b_4_true, t.is_b_5_true]
            true_same_as_lead = [t.same_as_lead_0_true, t.same_as_lead_1_true, t.same_as_lead_2_true, t.same_as_lead_3_true, t.same_as_lead_4_true]
            true_same_as_lead.insert(0,1)
            t1_list_true = [t for i,t in enumerate(jets_true) if true_same_as_lead[i]>0]
            #print("jets_true")
            #print(jets_true)
            #print("len(jets_true)")
            #print(len(jets_true))
            #print("t1_list_true")
            #print(t1_list_true)
            #print("true_same_as_lead")
            #print(true_same_as_lead)
            t1_true = t1_list_true[0]+t1_list_true[1]+t1_list_true[2]
            t2_list_true = [t for i,t in enumerate(jets_true) if true_same_as_lead[i]<1]
            t2_true = t2_list_true[0]+t2_list_true[1]+t2_list_true[2]
            # call t1 the top with leading pt
            if args.pt_order:
                if t2_true.Pt() > t1_true.Pt(): t1_true,t2_true=t2_true,t1_true          
            mt1_true[0] = t1_true.M()
            mt2_true[0] = t2_true.M()
        
            alpaca_good_mt1[0] = (round(mt1_true[0],2) == round(mt1_reco[0],2)) or (round(mt2_true[0],2) == round(mt1_reco[0],2))
            alpaca_good_mt2[0] = (round(mt2_true[0],2) == round(mt2_reco[0],2)) or  (round(mt1_true[0],2) == round(mt2_reco[0],2))
            
            '''
            if t.has_chi2:
                chi2_good_mt1[0] = (round(mt1_true[0],2) == round(mt1_chi2_nobfixed[0],2)) or (round(mt2_true[0],2) == round(mt1_chi2_nobfixed[0],2))
                chi2_good_mt2[0] = (round(mt2_true[0],2) == round(mt2_chi2_nobfixed[0],2)) or (round(mt1_true[0],2) == round(mt2_chi2_nobfixed[0],2)) 
            else:
            '''
            chi2_good_mt1[0] = -99
            chi2_good_mt2[0] = -99

        else:
            mt1_true[0] = -99
            mt2_true[0] = -99
            alpaca_good_mt1[0] = -99
            alpaca_good_mt2[0] = -99
            chi2_good_mt1[0] = -99
            chi2_good_mt2[0] = -99
            
        has_truth[0] = int(t.has_truth)

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

    if args.build_df:
        df = build_and_store_df(args, files_FT)
        to_root(df, args.merged_df_root, key=args.output_tree)

    m1, m2 = build_tree(args)
    #df['m1_chiara'] = m1
    #df['m2_chiara'] =  m2
    df.to_csv((args.output).replace('root','csv'))

if __name__=='__main__':
    main()

