ALPACA
======

ML project for pair-produced particle reconstruction.
The goal of this project is to identify decay patterns for e.g. ttbar or SUSY particle production and decays to many jets without iterating over the combinatorics.

The network is built in pytorch, and currently based on the combination/Lorentz layer structure introduced in [arXiv:1707.08966](https://arxiv.org/abs/1707.08966).
For now, it is set up to do top reconstruction.
Input files from AnalysisTop all-hadronic ntuples are generated with the *ttbarjetmatching* submodule.


Getting the code
----------------
Copy the gitlab link with your choice of protocol (ssh is usually good for personal computers, kerberos for cluster machines), then do
```
git clone --recursive [url]
```
to ensure that any submodules are checked out as well.


Installation
------------
Install with conda:
```
conda env create -f alpaca_conda.yml
```


Setup
-----
Just source `setup.sh`. The script will activate the conda environment created at the previous step and set a few environmental variables.


Start training!
---------------
See the [Run 2 RPV Multijets twiki page](https://twiki.cern.ch/twiki/bin/view/AtlasProtected/RpvMultiJetFullRun2) for a test file on cernbox. To run after setting up the environment, do in a working directory:
```
alpaca --sig truthmatched.h5
```
Logs and plots will be written to subfolders in `data/`. For more information you can call `alpaca --help`.


Package structure
-----------------
  - `alpaca/`: Contains modules defining different network architectures:
    - `simple.py`: Feed-forward network with a configurable number/size of hidden layers taking jet four-vectors as input
    - `colalola.py`: Lorentz-aware network with combination layer -- this first generates linear combinations of the four-vectors with trainable weights, then builds invariant masses and other Lorentz-invariants (some trainable) from the sums. The invariants are passed to a feed-forward network like the "simple" network. The number of combinations and the feed-forward hidden layer structure is configurable.
  - `bin/`: Contains runnable scripts, currently only `alpaca`, which executes the network configuration, training and performance evaluation. Could stand to be made more modular :)


Understanding the inputs and outputs
------------------------------------
### Input file
The input file is in *pandas* dataframe format structured as follows:
  - Rows --> events from an all-hadronic ttbar sample (Powheg+Pythia8)
  - Each row has `jet_px`, `jet_py`, `jet_pz` and `jet_e`, `partonindex` for up to 10 jets. If an event has fewer than 10 jets, the corresponding column values are zeroed out.

#### Parton index:
Each jet has a label based on delta-R matching to the truth partons from the top decay. The values are as follows:
  0. Not from the top decay
  1. b from top
  2. W1 from top
  3. W2 from top
  4. b from antitop
  5. W1 from antitop
  6. W2 from antitop

These are the values that in principle we want the network to understand and reconstruct from the event kinematics, with the caveat that we don't expect to be able to distinguish the charges of the top quarks, or those of the W decay products.

### Output format
The loss functions available in pytorch (in our understanding so far) are built around sigmoid output nodes that do binary classification. Our use case is closer to multi-class tagging, which is handled by outputting a discriminant for each possible class, such that the highest score identifies the most likely label.

We train the model to reproduce a series of 0/1 labels that encode the jet identities. One way to encode the full decay structure that we are interested in (ignoring all ambiguities) is:
  - N>6 top-jet discriminants: 1 = top jet, 0 = not from top decay ("ISR"). This is irrelevant if we only train the network on events with exactly 6 jets, all of which are from the top decays.
  - 5 top-matching discriminants: We try to assign the 6 jets from the top decays to two triplets corresponding to the two top quarks. WLOG we can try to find the other two jets that match the leading top jet, assigning 1 if the jet is from the same decay as the leading top jet, and 0 otherwise.
  - 6 b-jet discriminants: We try to use the kinematic information to tell apart the b-jets and the jets from the W decays, assigning 1 if the jet is a b-jet and 0 otherwise.

A concrete example is as follows:
```
Parton labels:    [ 0, 3, 4, 1, 5, 0, 2, 6, 0, 0 ]
Top jet labels:   [ 0, 1, 1, 1, 1, 0, 1, 1, 0, 0 ]
Top decay labels: [       0, 1, 0,    1, 0       ]
B-jet labels:     [    0, 1, 1, 0, 0, 0, 0       ]
```

After running the model on an event, one can therefore attempt to reconstruct the top system according to the following steps:
  1. Select the 6 jets with the highest top-jet score
  2. Of those, group together the leading top jet with the two other jets scoring highest in the top decay labels.
  3. The two highest scoring jets with the b-tag label are the b-jets.


To use alpaca with conda (after you setup your conda environment with named `alpaca`):
```bash
$ source setup.sh conda
$ alpaca -h
```

Usage example:

```bash
$ alpaca --sig /srv/beegfs/scratch/groups/dpnc/atlas/rpoggi/multijet/410471_6jexcl_2bjets_merged_mc_xs34.root --bkg /srv/beegfs/scratch/groups/dpnc/atlas/rpoggi/multijet/background_merged_a0b0_data_xs34.root
```

The results and a log file will be placed in an output directory (by default called `data`).


Contributors
------------
- Lukas Heinrich
- Riccardo Poggi
- Teng Jian Khoo
