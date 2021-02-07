# Jet-parton matching with uproot/pandas

import uproot, uproot_methods, math
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

folder = "/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/zoom_recordings/alpaca_tutorial/user.rpoggi.410471.PhPy8EG.DAOD_TOPQ1.e6337_e5984_s3126_r9364_r9315_p3629.TTDIFFXS36_R21_allhad_resolved.root/"

store_truth = True
select_truth = False
select_2bTag = True

#h5_name='/eos/user/c/crizzi/RPV/alpaca/h5/mio/truthmatched_btagging_2b_train_36_2_3_4.h5'
# Open root file
#fname = [folder+"*_00002*.allhad_resolved.root", folder+"*_00003*.allhad_resolved.root", folder+"*_00004*.allhad_resolved.root"]
#fname = [folder+"user.rpoggi.18378247.*root"]
#fname = [folder+"user.rpoggi.18378247.*00030*root"]

# h5_name='/eos/user/c/crizzi/RPV/alpaca/h5/mio/truthmatched_unfiltered_test_36_0_1.h5'
h5_name='/eos/user/c/crizzi/RPV/alpaca/h5/mio/truthmatched_btagging_2b_test_36_0_1_unfiltered.h5'
fname = [folder+"*_00000*.allhad_resolved.root", folder+"*_00001*.allhad_resolved.root"]

print("Working on file '{}'".format(fname))
# Define parton labels and corresponding indices
# 0 denotes a failure to match
partons = [ '{}_from_{}'.format(child,parent) for parent in ['t','tbar'] for child in ['b','Wdecay1','Wdecay2']]
partonindex = {parton:iparton for iparton,parton in enumerate([None]+partons)}

# Access trees as dataframe iterators reading multiple input files
print("Making pandas")
nom_iter = uproot.pandas.iterate(fname,"nominal",entrysteps=float("inf"),branches=["eventNumber","jet_*"])
truth_iter = uproot.pandas.iterate(fname,"truth",entrysteps=float("inf"),branches=["MC_b_from_t*","MC_Wdecay*_from_t*"])
# Set the truth df index so we can use "join"
# Join function seems to preserve subentry, which merge does not
truth_eventN_iter = uproot.pandas.iterate(fname,"truth",entrysteps=float("inf"),branches="eventNumber")

df_list=[]
ifile = 0
for nom_df, truth_df, truth_eventN_df in zip(nom_iter,truth_iter,truth_eventN_iter):
    print("Processing file {}".format(ifile))
    # Merge dataframes on "eventNumber" key
    # Merging truth into nominal retains only entries in nominal tree
    n_nom = nom_df.unstack(fill_value=0).shape[0]
    truth_df.index = truth_eventN_df["eventNumber"]
    truth_df['n_partons_truth'] = sum([truth_df["MC_"+p+"_m"]>0 for p in partons])
    #    truth_df['n_partons'] = truth_df.apply(lambda row: sum([row["MC_"+p+"_m"]>0 for p in partons]) )


    joined_df = nom_df.join(truth_df,on="eventNumber")
    joined_df = joined_df[joined_df["eventNumber"].notna()]

    # print(truth_df.head())
    # print(nom_df.head())

    print("  Computing dR")
    # Compute delta eta and delta phi
    twopi = 2*math.pi # mod takes precedence over times, so precompute
    for parton in partons:
        joined_df["jet_deta_{}".format(parton)] = joined_df["jet_eta"] - joined_df["MC_{}_eta".format(parton)]
        joined_df["jet_dphi_{}".format(parton)] = (joined_df["jet_phi"] - joined_df["MC_{}_phi".format(parton)] + math.pi) % twopi - math.pi
        joined_df["jet_dR_{}".format(parton)] = np.sqrt(joined_df["jet_deta_{}".format(parton)]**2 + joined_df["jet_dphi_{}".format(parton)]**2)    

    if store_truth:
        # Apply jet-parton matching
        print("  Matching partons to jets")
        dRbranches = ["jet_dR_{}".format(parton) for parton in partons]        
        match_dRcut = 0.4
        joined_df["partonlabel"] = np.where(joined_df[dRbranches].min(axis=1)<match_dRcut,joined_df[dRbranches].idxmin(axis=1).str.lstrip("jet_dR"),None)
        joined_df["partonindex"] = joined_df["partonlabel"].map(partonindex)
        
    # Define columns in cartesian coordinates, needed later for the network
    joined_df["jet_px"] = joined_df["jet_pt"]*np.cos(joined_df["jet_phi"])
    joined_df["jet_py"] = joined_df["jet_pt"]*np.sin(joined_df["jet_phi"])
    joined_df["jet_pz"] = joined_df["jet_pt"]*np.sinh(joined_df["jet_eta"])
    joined_df["jet_isDL177"] = joined_df["jet_DL1"]>1.45
        
    # Truncate the jet vector
    # maxjets = 10
    truncated_df = joined_df # joined_df[joined_df.index.get_level_values('subentry') < maxjets] # chiara: keep all jets
    truncated_df = truncated_df.unstack(fill_value=0)
    n_truncated = truncated_df.shape[0]
    
    if store_truth:
        def good_labels(df, ij=None): 
            if ij==None:
                gl = (
                    (np.count_nonzero(df['partonindex'] == 1, axis=1) == 1) &
                    (np.count_nonzero(df['partonindex'] == 2, axis=1) == 1) &
                    (np.count_nonzero(df['partonindex'] == 3, axis=1) == 1) &
                    (np.count_nonzero(df['partonindex'] == 4, axis=1) == 1) &
                    (np.count_nonzero(df['partonindex'] == 5, axis=1) == 1) &
                    (np.count_nonzero(df['partonindex'] == 6, axis=1) == 1)
                )
            else:
                gl = (
                    (np.count_nonzero(df['partonindex'][[i for i in range(ij)]] == 1, axis=1) == 1) &
                    (np.count_nonzero(df['partonindex'][[i for i in range(ij)]] == 2, axis=1) == 1) &
                    (np.count_nonzero(df['partonindex'][[i for i in range(ij)]] == 3, axis=1) == 1) &
                    (np.count_nonzero(df['partonindex'][[i for i in range(ij)]] == 4, axis=1) == 1) &
                    (np.count_nonzero(df['partonindex'][[i for i in range(ij)]] == 5, axis=1) == 1) &
                    (np.count_nonzero(df['partonindex'][[i for i in range(ij)]] == 6, axis=1) == 1)
                )
            return gl.astype(int)
       
        #print("truncated_df['partonindex']")
        #print(truncated_df['partonindex'][[i for i in range(3)]])
        truncated_df["n_matched",0] = np.count_nonzero(truncated_df['partonindex']>0 , axis=1)
        truncated_df["n_matched_6",0] = (truncated_df["n_matched",0]>5).astype(int)
        # print(truncated_df[truncated_df.n_matched>6][ dRbranches].head(50))
        truncated_df["n_partons",0] = truncated_df['n_partons_truth',0]
        for ij in [6,7,8,9,10,11,12]:
            truncated_df["good_match_"+str(ij)+"j",0] = good_labels(truncated_df,ij)
        truncated_df["good_match",0] = good_labels(truncated_df)
        #print('truncated_df["good_match",0]')
        #print(truncated_df["good_match",0])
        n_partonsexist = truncated_df[truncated_df['n_partons',0]>5].shape[0]
        n_atleastsixmatches = truncated_df[truncated_df['n_matched',0]>5].shape[0]
        n_sixmatches = truncated_df[truncated_df['n_matched',0]==6].shape[0]
        n_filtered =truncated_df[(truncated_df['n_matched',0]==6)&(truncated_df['n_partons',0]>5)].shape[0]

    truncated_df["event_number",0] = truncated_df["eventNumber",0]
    truncated_df["n_jets",0] = np.count_nonzero(truncated_df['jet_e']>0 , axis=1)
    truncated_df["n_DL177",0] = np.count_nonzero(truncated_df['jet_isDL177']>0 , axis=1)
        
    filtered_df = truncated_df
    if select_2bTag:
        filtered_df = filtered_df[ filtered_df["n_DL177",0]>1 ] 
    if select_truth:
        filtered_df = filtered_df[(filtered_df['n_matched',0]==6)&(filtered_df['n_partons',0]>5)]

    for ij in range(0,20):
        for col in ["partonindex"]+["jet_"+comp for comp in["e","px","py","pz","isDL177"]]:
            if not (col,ij) in filtered_df.columns:
                filtered_df[col,ij] = 0

    # Reshape the df to have one event per row,
    # filling each overflow entry with 0's
    # Extract just the cartesian coordinates and the parton index
    jet_cartesian = ["jet_"+comp for comp in ["e","px","py","pz","isDL177"]]
    # chiare: somehow some events become all nan
    filtered_df = filtered_df[filtered_df['eventNumber'].notna()]
    if store_truth:
        #filtered_df["partonindex"] = filtered_df["partonindex"].astype(int)            
        # unstacked_df = filtered_df[jet_cartesian+["partonindex","event_number","n_partons","n_matched","n_matched_6","good_match","n_jets","n_DL177"]+["good_match_"+str(ij)+"j" for ij in [6,7,8,9,10,11,12]]]
        unstacked_df = filtered_df[jet_cartesian+["partonindex","event_number","good_match","n_jets","n_DL177"]]
    else:
        unstacked_df = filtered_df[jet_cartesian+["event_number","n_DL177","n_jets"]]

    unstacked_df.fillna(0,inplace=True)
    #print(unstacked_df[["n_jets","n_partons","n_matched","n_matched_6","good_match"]+["good_match_"+str(ij)+"j" for ij in [6,7,8,9,10,11,12]]].describe())
    #print(unstacked_df.shape)
    #print(list(unstacked_df.columns))

    # Write out to an hdf5 file
    print("  Length of df I'm about to write: {}".format(len(filtered_df)), unstacked_df.shape)

    print("")
    print('Number of events in file', n_nom, float(n_nom)/n_nom)
    print('Number of events in truncated_df', n_truncated, float(n_truncated)/n_nom)
    if store_truth:
        print('Number of events with 6 partons', n_partonsexist, float(n_partonsexist)/n_nom)
        print('Number of events with at least 6 matched jets', n_atleastsixmatches, float(n_atleastsixmatches)/n_nom)
        print('Number of events with exactly 6 matched jets', n_sixmatches, float(n_sixmatches)/n_nom)
        print('Number of events with 6 partons and exactly 6 matched jets', n_filtered, float(n_filtered)/n_nom)
    print("Number of events I'm writing", unstacked_df.shape[0])
    print("")

    #print("  Writing to h5")
    #unstacked_df.to_hdf(h5_name,key="df",mode='w' if ifile==0 else 'a',append=True,complevel=0)
    #print("  Done processing {}".format(ifile))
    df_list.append(unstacked_df)
    ifile+=1

df=pd.concat(df_list)
df.to_hdf(h5_name,key="df",mode='w')


'''
print(unstacked_df.shape)
print(unstacked_df.columns)
print(unstacked_df.head(20))
print('\n Total')
print(unstacked_df[["n_jets","n_partons","n_matched","n_matched_6","good_match"]+["good_match_"+str(ij)+"j" for ij in [6,7,8,9,10,11,12]]].describe())
print('\n ==6j')
print(unstacked_df[unstacked_df.n_jets==6][["n_jets","n_partons","n_matched","n_matched_6","good_match"]+["good_match_"+str(ij)+"j" for ij in [6,7,8,9,10,11,12]]].describe())
print('\n ==7j')
print(unstacked_df[unstacked_df.n_jets==7][["n_jets","n_partons","n_matched","n_matched_6","good_match"]+["good_match_"+str(ij)+"j" for ij in [6,7,8,9,10,11,12]]].describe())
print('\n ==8j')
print(unstacked_df[unstacked_df.n_jets==8][["n_jets","n_partons","n_matched","n_matched_6","good_match"]+["good_match_"+str(ij)+"j" for ij in [6,7,8,9,10,11,12]]].describe())
print('\n ==9j')
print(unstacked_df[unstacked_df.n_jets==9][["n_jets","n_partons","n_matched","n_matched_6","good_match"]+["good_match_"+str(ij)+"j" for ij in [6,7,8,9,10,11,12]]].describe())
print('\n >=10j')
print(unstacked_df[unstacked_df.n_jets>9][["n_jets","n_partons","n_matched","n_matched_6","good_match"]+["good_match_"+str(ij)+"j" for ij in [6,7,8,9,10,11,12]]].describe())
'''
