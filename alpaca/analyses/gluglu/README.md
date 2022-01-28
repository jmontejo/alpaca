# Alpaca for the RPV MJ Analysis

## Technical setup

### First Time

If you don't already have it, install conda, e.g.:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

Please note that, on lxplus, the default location for the installation is your 
home directory. If you don't have enough space there (having a few environments can 
easily take a few GB), pay attention to the step of the installation where the destiation 
is specified and set it to something different than your home directory. 


After you have conda installed, you can create an environment with the necessary packages, as indicated in the main README of alpaca:
```
conda env create -f alpaca_conda.yml
```

### Every Time

Setup script:
```
source setup.sh
```
This script will also activate the conda environment. 

## Training and Evaluation

### Train for Gluino Reconstruction
 
The script [run_gluglu_hydra_3layers.sh](./scripts/run_gluglu_hydra_3layers.sh) contains the full workflow for the training and
evaluation for gluino reconstruction. In this example case, the training is performed on a single mass point.
This script performs:
* training
* evaluation on the training sample (to check overtraining)
* evaluation on the validation sample
* evaluation on different mass points and on the QCD sample

The main command is `alpaca gluglu` to run the `gluglu` analysis in alpaca, followed by the command line arguments.
The most important arguments are:
* `--input-file`: h5 file with the input
* `-t`: perform the training
* `-w`:
* `--hydra`: use the hydra architecture. Other options are `--simple-nn` and `--cola-lola`
* `--fflayer`: architecture of the feed forward layer. E.g. if 200 200 200, we'll have three layers with 200 nodes each. 
* `--test-sample`: how many events of that file should be used for the validation during the training.
The remaining events are used for the training. If set to `-1`, all of the events are used just for evaluation, and no training is performed
(used in the evaluation step)
* `--jets`: number of jets to be considered by alpaca for each event (usually 7 or 8)
* `--zero-jets`: how many of the previously specified number can be missing in the event (equal to `--jets` minus 6). E.g. if `--jets 8`, in the case of a gluino signal we can 
still reconstruct the gluino also with just 6 jets, so we would set `--zero-jets` to 2.
* `--spectators`: variables not used in the training but present in the h5 file and stored in the output file 
* `--scale-e`: scale the jet energy by the sum of the energy of all the jets in the event (tentative to make the training less mass dependent)
* `--output-dir`: where to place the results
* `--tag`: name of the training, will be the name of the folder in the result folder
* `--label-roc`: additional label for the ROC curve plot

Other options are visible in [cli.py](../../cli.py) (e.g. shuffle jets, per-jet training, ...).

### Train for Signal vs QCD Discrimination

In this case, the training is performed using both signal and background samples. 

## Visualize the Results





