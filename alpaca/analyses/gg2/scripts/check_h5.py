import pandas as pd
pd.set_option('max_columns', None)
import sys

print('hello')
#fname='/eos/user/c/crizzi/RPV/alpaca/h5/mio/truthmatched_gluino_truth_UDS_1400_all.h5'
fname='/eos/user/c/crizzi/RPV/alpaca/h5/mio/truthmatched_gluino_truth_UDS_1400_train.h5'
#fname='truthmatched_36.h5'
dfname='df'
df = pd.read_hdf(fname, dfname)
#df.to_csv("myh5intocsv.csv")
print(df.head())
#print("shape")
#print(df.shape)
#print("columns")
#print(list(df.columns))
sys.exit()
#sys.exit()
#print("head")
#print(df.head())
#print('parton index')
#print(df[['partonindex']].head(25))
#print("bjets n")
#print(df[['n_DL177']])

if False:
    df.to_csv("mytest.csv")
    n_total = df.shape[0]
    print('Total:', n_total)
    print('Not match loose:', df[df['good_match_loose',0]<1].shape[0])
    print('Less than 6 jets:', df[df['n_jets',0]<6].shape[0])
    
    to_investigate = df[(df['n_jets',0]>5) & (df['good_match_loose',0]<1)]
    print(to_investigate.shape[0])
    col_check = [c for c in to_investigate if 'from' in c[0]]
    to_investigate = to_investigate[col_check]
    print(col_check)
    print(to_investigate.head())

if True:
    df = df.iloc[:, df.columns.get_level_values(1)==0]
    print(df.head())
    print('\n Total')
    print(df[["n_jets","n_partons","n_matched","good_match","n_matched_6"]+["good_match_"+str(ij)+"j" for ij in [6,7,8,9,10,11,12]]].mean().to_frame().T)
    print('\n ==6j')
    print(df[df.n_jets==6][["n_jets","n_partons","n_matched","good_match","n_matched_6"]+["good_match_"+str(ij)+"j" for ij in [6,7,8,9,10,11,12]]].mean().to_frame().T)
    print('\n ==7j')
    print(df[df.n_jets==7][["n_jets","n_partons","n_matched","good_match","n_matched_6"]+["good_match_"+str(ij)+"j" for ij in [6,7,8,9,10,11,12]]].mean().to_frame().T)
    print('\n ==8j')
    print(df[df.n_jets==8][["n_jets","n_partons","n_matched","good_match","n_matched_6"]+["good_match_"+str(ij)+"j" for ij in [6,7,8,9,10,11,12]]].mean().to_frame().T)
    print('\n ==9j')
    print(df[df.n_jets==9][["n_jets","n_partons","n_matched","good_match","n_matched_6"]+["good_match_"+str(ij)+"j" for ij in [6,7,8,9,10,11,12]]].mean().to_frame().T)
    print('\n >=10j')
    print(df[df.n_jets>9][["n_jets","n_partons","n_matched","good_match","n_matched_6"]+["good_match_"+str(ij)+"j" for ij in [6,7,8,9,10,11,12]]].mean().to_frame().T)


