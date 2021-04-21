import pandas as pd
from itertools import chain, product

max_jets = 20
jq = ['jet_e','jet_px', 'jet_py', 'jet_pz', 'partonindex']
tup =[('entry',0)] + [(q,i) for q in jq for i in range(max_jets)]
tup.append(('event_number',0))
tup.append(('good_match',0))
tup.append(('n_jets',0))
print(tup)

#onindex,partonindex,partonindex,event_number,good_match,n_jets

infile = 'myh5intocsv.csv'
df = pd.read_csv(infile,skiprows=2,header=0, index_col =pd.MultiIndex.from_tuples(tup))
#df.columns = pd.MultiIndex.from_tuples(tup)
print(df.head())

print(df.columns)
