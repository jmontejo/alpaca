# Jet-parton matching with uproot/pandas

import uproot, uproot_methods, math
import numpy as np
import pandas as pd
import sys

pd.set_option('display.max_columns', None)

folder = '/eos/user/c/crizzi/RPV/ntuples/FT_signal_020321_merged/mc16e/signal/'

store_truth = True
select_truth = False

# h5_name='/eos/user/c/crizzi/RPV/alpaca/h5/mio/truthmatched_unfiltered_test_36_0_1.h5'
h5_name='/eos/user/c/crizzi/RPV/alpaca/h5/mio/truthmatched_gluino_UDB_1200.h5'
fname = [folder+'/504516.root']

print("Working on file '{}'".format(fname))
# Define parton labels and corresponding indices
# 0 denotes a failure to match
partons = [ '{}_from_{}'.format(child,parent) for parent in ['g1','g2'] for child in ['q1','q2','q3']]
partonindex = {parton:iparton for iparton,parton in enumerate([None]+partons)}

print(partons)

# Access trees as dataframe iterators reading multiple input files
print('Making pandas')
nom_iter = uproot.pandas.iterate(fname,'trees_SRRPV_',entrysteps=float('inf'),branches=['eventNumber','jet_eta','jet_phi','jet_pt','jet_e'])
truth_iter = uproot.pandas.iterate(fname,'trees_SRRPV_',entrysteps=float('inf'),branches=['eventNumber','truth_QuarkFromGluino_phi','truth_QuarkFromGluino_eta','truth_QuarkFromGluino_pt'])#, 'truth_QuarkFromGluino_ParentBarcode'])

df_list=[]
ifile = 0
for nom_df,truth_df  in zip(nom_iter,truth_iter):
    print('Processing file {}'.format(ifile))
    # Merge dataframes on 'eventNumber' key
    # Merging truth into nominal retains only entries in nominal tree
    n_nom = nom_df.unstack(fill_value=0).shape[0]
    # print(nom_df.head())
    print(truth_df.head())
    col_orig = truth_df.columns.get_level_values(0) 
    truth_df = truth_df.unstack(level=-1)
    col_new = [c+'_'+str(n) for c in col_orig for n in range(6)]
    #print(col_new)
    #print(truth_df.columns)
    #print(truth_df.columns.droplevel(0))
    truth_df.columns = col_new
    truth_df['eventNumber'] = truth_df['eventNumber_0']
    truth_df.index = truth_df['eventNumber']
    to_keep = [c for c in truth_df.columns if 'truth' in c]
    truth_df = truth_df[to_keep]
    col_new = [c+'_'+v for v in ['phi','eta','pt'] for c in partons]
    print(truth_df.head())
    print(truth_df.columns)
    print(col_new)
    truth_df.columns = col_new
    truth_df['n_partons_truth'] = sum([truth_df[p+'_phi']*truth_df[p+'_phi']>0 for p in partons])

    joined_df = nom_df.join(truth_df,on='eventNumber')

    print('  Computing dR')
    # Compute delta eta and delta phi
    twopi = 2*math.pi # mod takes precedence over times, so precompute
    for parton in partons:
        joined_df['jet_deta_{}'.format(parton)] = joined_df['jet_eta'] - joined_df['{}_eta'.format(parton)]
        joined_df['jet_dphi_{}'.format(parton)] = (joined_df['jet_phi'] - joined_df['{}_phi'.format(parton)] + math.pi) % twopi - math.pi
        joined_df['jet_dR_{}'.format(parton)] = np.sqrt(joined_df['jet_deta_{}'.format(parton)]**2 + joined_df['jet_dphi_{}'.format(parton)]**2)    

    # chiara: add loop to compute dR between partons and store minimum

    if store_truth:
        # Apply jet-parton matching
        print('  Matching partons to jets')
        dRbranches = ['jet_dR_{}'.format(parton) for parton in partons]        
        match_dRcut = 0.4
        joined_df['partonlabel'] = np.where(joined_df[dRbranches].min(axis=1)<match_dRcut,joined_df[dRbranches].idxmin(axis=1).str.lstrip('jet_dR'),None)
        joined_df['partonindex'] = joined_df['partonlabel'].map(partonindex)
        
    # Define columns in cartesian coordinates, needed later for the network
    joined_df['jet_px'] = joined_df['jet_pt']*np.cos(joined_df['jet_phi'])
    joined_df['jet_py'] = joined_df['jet_pt']*np.sin(joined_df['jet_phi'])
    joined_df['jet_pz'] = joined_df['jet_pt']*np.sinh(joined_df['jet_eta'])
    # joined_df['jet_isDL177'] = joined_df['jet_DL1']>1.45
        
    # Truncate the jet vector
    maxjets = 20 # chiara: change back to keep all jets
    # truncated_df = joined_df # chiara: keep all jets
    truncated_df = joined_df[joined_df.index.get_level_values('subentry') < maxjets] # chiara: keep all jets
    truncated_df = truncated_df.unstack(fill_value=0)
    n_truncated = truncated_df.shape[0]

    print(partonindex)
    
    if store_truth:
        def good_labels(df, ij=None, loose=False): 
            if ij==None:
                if loose:
                    gl = (
                        (np.count_nonzero(df['partonindex'] == 1, axis=1) >= 1) &
                        (np.count_nonzero(df['partonindex'] == 2, axis=1) >= 1) &
                        (np.count_nonzero(df['partonindex'] == 3, axis=1) >= 1) &
                        (np.count_nonzero(df['partonindex'] == 4, axis=1) >= 1) &
                        (np.count_nonzero(df['partonindex'] == 5, axis=1) >= 1) &
                        (np.count_nonzero(df['partonindex'] == 6, axis=1) >= 1)
                    )
                else:
                    gl = (
                        (np.count_nonzero(df['partonindex'] == 1, axis=1) == 1) &
                        (np.count_nonzero(df['partonindex'] == 2, axis=1) == 1) &
                        (np.count_nonzero(df['partonindex'] == 3, axis=1) == 1) &
                        (np.count_nonzero(df['partonindex'] == 4, axis=1) == 1) &
                        (np.count_nonzero(df['partonindex'] == 5, axis=1) == 1) &
                        (np.count_nonzero(df['partonindex'] == 6, axis=1) == 1)
                    )
            else:
                if loose:
                    gl = (
                        (np.count_nonzero(df['partonindex'][[i for i in range(ij)]] == 1, axis=1) >= 1) &
                        (np.count_nonzero(df['partonindex'][[i for i in range(ij)]] == 2, axis=1) >= 1) &
                        (np.count_nonzero(df['partonindex'][[i for i in range(ij)]] == 3, axis=1) >= 1) &
                        (np.count_nonzero(df['partonindex'][[i for i in range(ij)]] == 4, axis=1) >= 1) &
                        (np.count_nonzero(df['partonindex'][[i for i in range(ij)]] == 5, axis=1) >= 1) &
                        (np.count_nonzero(df['partonindex'][[i for i in range(ij)]] == 6, axis=1) >= 1)
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
       
        #print('truncated_df['partonindex']')
        #print(truncated_df['partonindex'][[i for i in range(3)]])
        truncated_df['n_matched',0] = np.count_nonzero(truncated_df['partonindex']>0 , axis=1)
        truncated_df['n_matched_6',0] = (truncated_df['n_matched',0]>5).astype(int)
        # print(truncated_df[truncated_df.n_matched>6][ dRbranches].head(50))
        truncated_df['n_partons',0] = truncated_df['n_partons_truth',0]
        for ij in [6,7,8,9,10,11,12]:
            truncated_df['good_match_'+str(ij)+'j',0] = good_labels(truncated_df,ij)
            truncated_df['good_match_loose_'+str(ij)+'j',0] = good_labels(truncated_df,ij, loose=True)
        truncated_df['good_match',0] = good_labels(truncated_df)
        truncated_df['good_match_loose',0] = good_labels(truncated_df, loose=True)
        #print('truncated_df['good_match',0]')
        #print(truncated_df['good_match',0])
        n_partonsexist = truncated_df[truncated_df['n_partons',0]>5].shape[0]
        n_atleastsixmatches = truncated_df[truncated_df['n_matched',0]>5].shape[0]
        print('chiara: not matched')
        print(truncated_df[truncated_df['n_matched',0]<=5].head(10))
        n_sixmatches = truncated_df[truncated_df['n_matched',0]==6].shape[0]
        n_filtered =truncated_df[(truncated_df['n_matched',0]==6)&(truncated_df['n_partons',0]>5)].shape[0]
        n_good_match = truncated_df[truncated_df['good_match',0]>0].shape[0]
        n_good_match_loose = truncated_df[truncated_df['good_match_loose',0]>0].shape[0]

    truncated_df['event_number',0] = truncated_df['eventNumber',0]
    truncated_df['n_jets',0] = np.count_nonzero(truncated_df['jet_e']>0 , axis=1)
    #truncated_df['n_DL177',0] = np.count_nonzero(truncated_df['jet_isDL177']>0 , axis=1)
        
    filtered_df = truncated_df
    if select_truth:
        filtered_df = filtered_df[(filtered_df['n_matched',0]==6)&(filtered_df['n_partons',0]>5)]

    for ij in range(0,20):
        for col in ['partonindex']+['jet_'+comp for comp in['e','px','py','pz']]:
            if not (col,ij) in filtered_df.columns:
                filtered_df[col,ij] = 0

    # Reshape the df to have one event per row,
    # filling each overflow entry with 0's
    # Extract just the cartesian coordinates and the parton index
    jet_cartesian = ['jet_'+comp for comp in ['e','px','py','pz']]
    truth_q_info = [p+'_'+v for p in partons for v in ['phi','eta','pt']]
    print('chiara, col', list(filtered_df.columns))
    # chiare: somehow some events become all nan
    filtered_df = filtered_df[filtered_df['eventNumber'].notna()]
    if store_truth:
        #filtered_df['partonindex'] = filtered_df['partonindex'].astype(int)            
        unstacked_df = filtered_df[jet_cartesian+truth_q_info+['partonindex','event_number','n_partons','n_matched','n_matched_6','good_match','n_jets','good_match_loose']+['good_match_'+str(ij)+'j' for ij in [6,7,8,9,10,11,12]]]
        unstacked_df = unstacked_df.iloc[:, unstacked_df.columns.get_level_values(1)==0]
        #unstacked_df = filtered_df[jet_cartesian+['partonindex','event_number','good_match','n_jets']]
    else:
        unstacked_df = filtered_df[jet_cartesian+['event_number','n_jets']]

    unstacked_df.fillna(0,inplace=True)
    #print(unstacked_df[['n_jets','n_partons','n_matched','n_matched_6','good_match']+['good_match_'+str(ij)+'j' for ij in [6,7,8,9,10,11,12]]].describe())
    #print(unstacked_df.shape)
    #print(list(unstacked_df.columns))

    # Write out to an hdf5 file
    print("  Length of df I'm about to write: {}".format(len(filtered_df)), unstacked_df.shape)

    print('')
    print('Number of events in file', n_nom, float(n_nom)/n_nom)
    print('Number of events in truncated_df', n_truncated, float(n_truncated)/n_nom)
    if store_truth:
        print('Number of events with 6 partons', n_partonsexist, float(n_partonsexist)/n_nom)
        print('Number of events with at least 6 matched jets', n_atleastsixmatches, float(n_atleastsixmatches)/n_nom)
        print('Number of events with exactly 6 matched jets', n_sixmatches, float(n_sixmatches)/n_nom)
        print('Number of events with 6 partons and exactly 6 matched jets', n_filtered, float(n_filtered)/n_nom)
        print('Number of events with good match', n_good_match, float(n_good_match)/n_nom)
        print('Number of events with good match loose', n_good_match_loose, float(n_good_match_loose)/n_nom)
    print('Number of events I am writing', unstacked_df.shape[0])
    print('')

    #print('  Writing to h5')
    #unstacked_df.to_hdf(h5_name,key='df',mode='w' if ifile==0 else 'a',append=True,complevel=0)
    #print('  Done processing {}'.format(ifile))
    df_list.append(unstacked_df)
    ifile+=1

df=pd.concat(df_list)
df.to_hdf(h5_name,key='df',mode='w')



#print(unstacked_df.shape)
#print(unstacked_df.columns)
#print(unstacked_df.head(20))
'''
print('\n Total')
print(unstacked_df[['n_jets','n_partons','n_matched','n_matched_6','good_match']+['good_match_'+str(ij)+'j' for ij in [6,7,8,9,10,11,12]]].describe())
print('\n ==6j')
print(unstacked_df[unstacked_df.n_jets==6][['n_jets','n_partons','n_matched','n_matched_6','good_match']+['good_match_'+str(ij)+'j' for ij in [6,7,8,9,10,11,12]]].describe())
print('\n ==7j')
print(unstacked_df[unstacked_df.n_jets==7][['n_jets','n_partons','n_matched','n_matched_6','good_match']+['good_match_'+str(ij)+'j' for ij in [6,7,8,9,10,11,12]]].describe())
print('\n ==8j')
print(unstacked_df[unstacked_df.n_jets==8][['n_jets','n_partons','n_matched','n_matched_6','good_match']+['good_match_'+str(ij)+'j' for ij in [6,7,8,9,10,11,12]]].describe())
print('\n ==9j')
print(unstacked_df[unstacked_df.n_jets==9][['n_jets','n_partons','n_matched','n_matched_6','good_match']+['good_match_'+str(ij)+'j' for ij in [6,7,8,9,10,11,12]]].describe())
print('\n >=10j')
print(unstacked_df[unstacked_df.n_jets>9][['n_jets','n_partons','n_matched','n_matched_6','good_match']+['good_match_'+str(ij)+'j' for ij in [6,7,8,9,10,11,12]]].describe())
'''
