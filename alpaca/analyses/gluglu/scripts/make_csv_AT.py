import pandas as pd
import numpy as np 
from root_pandas import to_root
import ROOT
from array import array
import uproot
import progressbar
import argparse



def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output',     default='outtree_nopTchoice.root',        help="Name of output ROOT file")
    return parser.parse_args()


def get_chi2_df(files_chi2, suff='', only_mass=False):
    df_chi2_list=[]
    print('Loading ROOT files')
    for f in  progressbar.progressbar(files_chi2):
        tname_chi2 = 'nominal'
        f_chi2 = uproot.open(f)
        t_chi2 = f_chi2[tname_chi2]
        df_chi2_list.append(t_chi2.pandas.df(['eventNumber', 'reco_Chi2Fitted', 'reco_t1_m', 'reco_t2_m', 'jet_pt*','jet_eta','jet_phi','reco_bjets','reco_DRbb','reco_DRbWMax'], flatten=False))

    df_chi2 = pd.concat(df_chi2_list)    
    df_chi2['pt_jet_0'] = df_chi2['jet_pt'].apply(lambda x: x[0])
    df_chi2['pt_jet_1'] = df_chi2['jet_pt'].apply(lambda x: x[1])
    df_chi2['pt_jet_2'] = df_chi2['jet_pt'].apply(lambda x: x[2])
    df_chi2['eta_jet_0'] = df_chi2['jet_eta'].apply(lambda x: x[0])
    df_chi2['eta_jet_1'] = df_chi2['jet_eta'].apply(lambda x: x[1])
    df_chi2['eta_jet_2'] = df_chi2['jet_eta'].apply(lambda x: x[2])
    df_chi2['phi_jet_0'] = df_chi2['jet_phi'].apply(lambda x: x[0])
    df_chi2['phi_jet_1'] = df_chi2['jet_phi'].apply(lambda x: x[1])
    df_chi2['phi_jet_2'] = df_chi2['jet_phi'].apply(lambda x: x[2])
    df_chi2['njets'] = df_chi2['jet_pt'].apply(lambda x: len(x))
    df_chi2['njets_55'] = df_chi2['jet_pt'].apply(lambda x: len([j for j in x if j > 55000]))
    df_chi2['njets_25'] = df_chi2['jet_pt'].apply(lambda x: len([j for j in x if j > 25000]))
    # selections from all had ttbar analysis, except the mass selection
    df_chi2['pass_reco_chi2'] = np.where((df_chi2['reco_Chi2Fitted']<10), 1, 0)
    df_chi2['pass_anatop'] = np.where((df_chi2['njets_55']>5) & (df_chi2['reco_bjets']==2)  & (df_chi2['reco_DRbb']>2) & (df_chi2['reco_DRbWMax']<2.2), 1, 0)
    if only_mass:
        df_chi2 = df_chi2[['eventNumber', 'reco_Chi2Fitted', 'reco_t1_m', 'reco_t2_m', 'pass_reco_chi2']]
    else:
        df_chi2 = df_chi2[['eventNumber', 'reco_Chi2Fitted', 'reco_t1_m', 'reco_t2_m', 'njets', 'njets_55','njets_25','pass_reco_chi2','pass_anatop','pt_jet_1','pt_jet_2','pt_jet_0','eta_jet_1', 'eta_jet_2', 'eta_jet_0', 'phi_jet_1', 'phi_jet_2', 'phi_jet_0']]
    if len(suff)>0:
        col_name = [c+'_'+suff if 'reco' in c else c for c in df_chi2.columns]
        df_chi2.columns = col_name
    return df_chi2


def build_and_store_df(args, files_chi2_nobfixed, files_chi2_bfixed):
    df_to_concat=[]

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
    
    df = pd.merge(df_chi2_bfixed, df_chi2_nobfixed, left_on='eventNumber', right_on='eventNumber', how='inner')
    print(df.shape)
    df = df.drop_duplicates(subset=['eventNumber'])
    print('df after drop duplicates')
    print(df.shape)
    print('null elements:',df.isnull().sum().sum())
    print(df.head())
    df.to_csv(args.output)
    return df
    

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
                           "user.rpoggi.18378247._000019.allhad_resolved.root",
                           "user.rpoggi.18378247._000020.allhad_resolved.root", 
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

    df = build_and_store_df(args, files_chi2_nobfixed, files_chi2_bfixed)

if __name__=='__main__':
    main()

