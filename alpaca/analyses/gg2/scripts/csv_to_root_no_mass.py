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
    parser.add_argument('--input-alpaca',     default='NNoutput_test.csv',        help="Input csv from alpaca with truth info")
    parser.add_argument('--merged-df',     default='merged_df_test.csv',        help="Merged df name")
    parser.add_argument('--merged-df-root',     default='merged_df_test.root',        help="Merged df name when stored as TTree")
    parser.add_argument('--output-tree',     default='tree',        help="Merged df name")
    parser.add_argument('--build-df',    action='store_true' ,       help="Create the merged dataframe and store it as root file")
    parser.add_argument('--no-input-root',    action='store_true' ,       help="Do not add the information in the original ROOT file")
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

    df_alpaca = pd.read_csv(args.input_alpaca)
    #columns_reco = [c for c in df_alpaca_noTruth.columns if 'true' not in c]
    #df_alpaca = df_alpaca_noTruth[columns_reco]
    df_alpaca['jet_0_round'] = np.sqrt( df_alpaca['jet_px_0']**2 + df_alpaca['jet_py_0']**2  ).round(2)
    print('df_alpaca')
    print(df_alpaca.shape)
    print(df_alpaca.columns)
    #print(df_alpaca.head())

    df_alpaca.fillna(0, inplace=True) # put the truth info at zero where missing

    print('df_alpaca after fillna')
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

    f_out = ROOT.TFile.Open(args.output,'RECREATE')
    t_out = ROOT.TTree(args.output_tree, args.output_tree)
 
    score = array('d',[0])
    ht = array('d',[0])
    ht_50 = array('d',[0])
    ht_8j = array('d',[0])
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

    wei  = array('d',[0])

    t_out.Branch("score",  score, "score/D")
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

    t_out.Branch("weight",  wei, "weight/D")

    f=ROOT.TFile.Open(args.merged_df_root,'READ')
    t=f.Get(args.output_tree)


    nentries=t.GetEntries()
    #nentries=1000 # chiara: tmp test
    for ientry in progressbar.progressbar(range(nentries)):
    # for ientry in range(5):
        t.GetEntry(ientry)

        score[0] = t.tagged

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
                    
        wei[0]=1
        for w in args.weights: 
            readw = getattr(t, w)
            wei[0] = wei[0]*readw

        t_out.Fill()
        
        #if ientry%1000==0:
        #    print(ientry)
        #    print('M1 reco:',t1.M(),      ' M2 reco:',t2.M())
        #    print('M1 true:',t1_true.M(), ' M2 true:',t2_true.M())
        
    f_out.Write()
    f_out.Close()
    f.Close()
    

def main():
    args = options()

    if args.build_df:
        df = build_and_store_df(args, files_FT)
        to_root(df, args.merged_df_root, key=args.output_tree)

    build_tree(args)
    df.to_csv((args.output).replace('root','csv'))

if __name__=='__main__':
    main()

