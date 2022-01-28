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

def main():
    d = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data=d)
    print("df")
    print(df.shape)
    print(df)
    to_root(df, "test.root"  , "tree")
    #df.to_csv("test.csv")

if __name__=='__main__':
    main()

