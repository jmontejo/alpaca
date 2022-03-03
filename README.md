ALPACA
======

ML project for event reconstruction.
The original goal of this project is to identify decay patterns for e.g. ttbar or SUSY particle production and decays to several (potentially different) decay products without iterating over the combinatorics. 
The framework also provides the possibility to perform a signal vs background classification. 

The network is built in pytorch. 
Several network architectures are available, in particular:
- a simple feed-forward network
- the CoLaLoLa architecture based on the combination/Lorentz layer structure introduced in [arXiv:1707.08966](https://arxiv.org/abs/1707.08966)
- the Hydra architecture, where ColaLola is used to first identify ISR jets and afterwards to reconstruct the event

Getting the code
----------------
Copy the gitlab link with your choice of protocol (ssh is usually good for personal computers, kerberos for cluster machines), then do
```
git clone --recursive [url]
```
to ensure that any submodules are checked out as well.


Environment
------------
Setup your environment with conda:
```
conda env create -f alpaca_conda.yml
```
Or make sure you have installed all the packages listed in `alpaca_conda.yml` under `dependencies`. 


Setup
-----
Just source `setup.sh`, the script will activate the conda environment created at the previous step and set a few environmental variables.
```
source setup.sh
```

Usage example
-------------
Each analysis has its own alpaca sub-command. The specific command, therefore, changes from analysis to analysis. 
This is also under active development, so the README might easily get out of sync with the actual code. 
The geenral structure of the command is: 
```
alpaca <analysis-name> <analysis options>
```
To have a list of all the implemented analyses:
```
alpaca --help`.
```
And, in order see all the commands available for each analysis:
```
alpaca <analysis-name> --help
```
Here's an example of usage for the SUSY RPV MJ analysis:
```
alpaca gluglu -t --input-file /eos/user/c/crizzi/RPV/alpaca/h5/mio/FT_022621/truthmatched_gluino_truth_UDS_1400_mymatch_train.h5 --test-sample 1000 -w  --tag alpaca_test --output-dir /eos/user/c/crizzi/RPV/alpaca/results/ --hydra --fflayer 200 200 200 --zero-jets 2 --jets 8  --spectators n_jets --spectators event_number --spectators weight --spectators has_good_match --extra-jet-fields jet_isSig --extra-jet-fields jet_isGluon
```

Analysis-Specific Instructions
-----------------
* RPV Multijet: [here](alpaca/analyses/gluglu)

Package structure
-----------------
  - `alpaca/nn/`: Contains modules defining different network architectures, e.g.:
    - `simple.py`: Feed-forward network with a configurable number/size of hidden layers taking jet four-vectors as input
    - `colalola.py`: Lorentz-aware network with combination layer -- this first generates linear combinations of the four-vectors with trainable weights, then builds invariant masses and other Lorentz-invariants (some trainable) from the sums. The invariants are passed to a feed-forward network like the "simple" network. The number of combinations and the feed-forward hidden layer structure is configurable.
  - `alpaca/analyses/`: Contains modules which implement your specific analysis sub-command, and scripts specific to each analysis
  - `bin/`: Contains important executables that are added to your `$PATH`, currently only `alpaca`, which executes the network configuration, training and performance evaluation. Could stand to be made more modular :)
  - `scripts/`: Contains utility scripts.


Example of input and output in the case of all-hadronic ttbar
------------------------------------
#### Input file
The input file is in *pandas* dataframe format structured as follows:
  - Rows --> events 
  - Each row has the per-event and per-jet information

##### Example of input file in the case of the all-hadronic ttbar analysis
  - Rows --> events from an all-hadronic ttbar sample (Powheg+Pythia8)
  - Each row has `jet_px`, `jet_py`, `jet_pz` and `jet_e`, `partonindex` for up to 10 jets. If an event has fewer than 10 jets, the corresponding column values are zeroed out.

###### Parton index:
Each jet has a label based on delta-R matching to the truth partons from the top decay. The values are as follows:
  0. Not from the top decay
  1. b from top
  2. W1 from top
  3. W2 from top
  4. b from antitop
  5. W1 from antitop
  6. W2 from antitop

These are the values that in principle we want the network to understand and reconstruct from the event kinematics, with the caveat that we don't expect to be able to distinguish the charges of the top quarks, or those of the W decay products.

#### Output

The loss functions available in pytorch (in our understanding so far) are built around sigmoid output nodes that do binary classification. Our use case is closer to multi-class tagging, which is handled by outputting a discriminant for each possible class, such that the highest score identifies the most likely label.
Since the output format can be analysis-specific, it needs to be implemented in each analysis through the function `write_output`. 
An example for the RPV MJ analysis is available [here](https://gitlab.cern.ch/atlas-phys-susy-wg/RPVLL/rpvmultijet/alpaca/-/blob/feature/analyses/alpaca/analyses/gluglu/__init__.py#L49). 

##### Output format in the case of all-hadronic ttbar analysis

We train the model to reproduce a series of 0/1 labels that encode the jet identities. One way to encode the full decay structure that we are interested in (ignoring all ambiguities) is:
  - N>6 top-jet discriminants: 1 = top jet, 0 = not from top decay ("ISR"). This is irrelevant if we only train the network on events with exactly 6 jets, all of which are from the top decays.
  - 5 top-matching discriminants: We try to assign the 6 jets from the top decays to two triplets corresponding to the two top quarks. WLOG we can try to find the other two jets that match the leading top jet, assigning 1 if the jet is from the same decay as the leading top jet, and 0 otherwise.
  - 6 b-jet discriminants: We try to use the kinematic information to tell apart the b-jets and the jets from the W decays, assigning 1 if the jet is a b-jet and 0 otherwise.

A concrete example is as follows:
```
Parton labels:    [ 0, 3, 4, 1, 5, 0, 2, 6, 0, 0 ]
Top jet labels:   [ 0, 1, 1, 1, 1, 0, 1, 1, 0, 0 ]
Top decay labels: [       0, 1, 0,    1, 0       ]
B-jet labels:     [    0, 1, 1, 0,    0, 0       ]
```

After running the model on an event, one can therefore attempt to reconstruct the top system according to the following steps:
  1. Select the 6 jets with the highest top-jet score
  2. Of those, group together the leading top jet with the two other jets scoring highest in the top decay labels.
  3. The two highest scoring jets with the b-tag label are the b-jets.


Contributors
------------
You can check the alpaca activities [here](https://gitlab.cern.ch/atlas-phys-susy-wg/RPVLL/rpvmultijet/alpaca/activity).