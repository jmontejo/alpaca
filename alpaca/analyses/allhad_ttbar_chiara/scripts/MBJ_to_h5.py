# Jet-parton matching with uproot/pandas

import uproot, uproot_methods, math
import numpy as np
import pandas as pd

# Open root file
quick_test = False
only_100k = True
fname = '/eos/user/c/crizzi/RPV/ntuples/21.2.87-1_SUSY4EWK-1/merged/Bkgtest_skim.root'
tname = 'dijet_nominal'
name_out = "/eos/user/c/crizzi/RPV/alpaca/h5/mio/dijet_MBJ"

# fname = '/eos/user/e/eantipov/Files/MBJ_21.2.87-1_1/3b/Bkg_ttbar_mc16e_21.2.87-1_output_histfitter.root'
# tname = 'ttbar_nominal'
# name_out = "/eos/user/c/crizzi/RPV/alpaca/h5/mio/ttbar_non_allhad"

spectators = ['event_number','channel_number','weight_mc','weight_lumi']


if only_100k:
    name_out = name_out+'_small'
name_out = name_out+'.h5'

if quick_test:
    fname = '/eos/user/c/crizzi/RPV/ntuples/21.2.87-1_SUSY4EWK-1/merged/small_test.root'
    tname = 'nominal'
print("Working on file '{}'".format(fname))
# Access trees as dataframe iterators reading multiple input files
print("Making pandas")

nsubentries = 10
columns_to_read = ['pt','eta','phi','m']
columns_to_read = ['small_R_jets_'+v+'_'+str(i) for v in columns_to_read for i in range(10) ]
columns_to_read += spectators
nom_iter = uproot.pandas.iterate(fname,tname,entrysteps=float("inf"),branches=columns_to_read)

ifile = 0
for nom_df  in nom_iter:
    print("Processing file {}".format(ifile))
    print(nom_df.shape)

    if only_100k:
        print('store only 100k events')
        nom_df = nom_df.sample(frac=1).reset_index(drop=True)
        nom_df = nom_df.loc[0:80000,:]
        print('done selecting 100k events')
        print(nom_df.head())
        print(nom_df['channel_number'].head())


    print('add event number for each jet')
    for i in range(10):
        for s in spectators:
            nom_df[s+'_'+str(i)]=nom_df[s]

    print(nom_df.shape)
    print(nom_df.head())
    print(nom_df[[c for c in nom_df.columns if '_0' in c]].head())
    
    print('change pT and m to MeV')
    for c in nom_df.columns:
        if ('small_R_jets_pt' in c) or ('small_R_jets_m' in c):
            nom_df[c] =  nom_df[c] * 1000.

    print(nom_df.shape)
    print(nom_df.head())
    print(nom_df[[c for c in nom_df.columns if '_0' in c]].head())

    print('compute quantities')
    for i in range(nsubentries):
        print('  ',i)
        nom_df["jet_px_"+str(i)] = nom_df["small_R_jets_pt_"+str(i)]*np.cos(nom_df["small_R_jets_phi_"+str(i)])
        nom_df["jet_py_"+str(i)] = nom_df["small_R_jets_pt_"+str(i)]*np.sin(nom_df["small_R_jets_phi_"+str(i)])
        nom_df["jet_pz_"+str(i)] = nom_df["small_R_jets_pt_"+str(i)]*np.sinh(nom_df["small_R_jets_eta_"+str(i)])
        nom_df["jet_p2_"+str(i)] = nom_df["jet_px_"+str(i)]*nom_df["jet_px_"+str(i)] + nom_df["jet_py_"+str(i)]*nom_df["jet_py_"+str(i)] + nom_df["jet_pz_"+str(i)]*nom_df["jet_pz_"+str(i)]
        nom_df["jet_e_"+str(i)] = np.sqrt ( nom_df["jet_p2_"+str(i)] + (nom_df["small_R_jets_m_"+str(i)]*nom_df["small_R_jets_m_"+str(i)]) )

    var = ['jet_e','jet_px','jet_py','jet_pz']+spectators
    columns_keep = [v+'_'+str(i) for v in var for i in range(nsubentries)]
    print('keep only important columns')
    print(columns_keep)
    print(nom_df.head())
    nom_df = nom_df[columns_keep]
    print(nom_df.shape)
    print('reindex')
    subentries = [i for i in range(nsubentries)]
    col_new = pd.MultiIndex.from_product([var, subentries], names=[None, 'subentry'])
    print(col_new)
    nom_df.columns = col_new
    print(nom_df.head(20))

    print("  Writing to h5")
    print(nom_df.shape)        
    '''
    iwrite =0
    for iiter in range(10):
        print('writing', iiter*680116)
        nom_df.loc[iiter*680116:(iiter+1)*680116 ,:].to_hdf("/eos/user/c/crizzi/RPV/test_fromMBJ.h5",key="df",mode='w' if (ifile==0) else 'a',append=True)
        iwrite+=1
    '''
    nom_df.to_hdf(name_out, key="df",mode='w' if (ifile==0) else 'a',append=True)
    print("  Done processing {}".format(ifile))

    ifile+=1
    
