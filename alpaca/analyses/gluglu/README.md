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
* `--input-file`, `-i`: h5 file with the input. **Note**: it can be called several times to consider multiple files, e.g. in the case of training on multiple
masses simultaneously. 
* `--train`, `-t`: perform the training
* `--write-output`, `-w`: store the result of the evaluation on the test sample
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
An interesting option not used in the example script is `--extra-jet-fields`, to include in the training extra per-jet variables (e.g. quark-gluon tagging, b-tagging, ...).

### Train for Signal vs QCD Discrimination

The same architectures that can be used for gluino reconstructions, can be used also for signal vs background training.
In the code, this is corresponds simply to changing the structure of the target vector. From the command line, this can
be achieved simply by including several input files, and adding a cagory for each (this is done with the option `--input-categories`, `-ic`).
E.g.:
```
alpaca gluglu -i signal.h5 -ic 1 -i qcd.h5 -ic 0 ... 
```
Examples of this usage are presented in the scripts [run_gluglu_sig_bkg_NN.sh](./scripts/run_gluglu_sig_bkg_NN.sh)
and [run_gluglu_sig_bkg.sh](./scripts/run_gluglu_sig_bkg.sh), that use a simple NN and the cola-lola architecture respectively 

In this case, the training is performed using both signal and background samples. 

## Visualize the Results

Note on the technical setup: On top of the packages needed to run alpaca, this steps requires also ROOT.
You can add the needed packages to your environment:
```
conda install -c conda-forge root
conda install -c conda-forge root_pandas
```
This needs to be done only once. 

The sequence of steps needed to visualize the results is reproduced in the script [streamline_gluino.sh](./scripts/streamline_gluino.sh).
This script takes as input the number of jets used by alpaca and the name of the tag, while other important parameters that are less frequently changed
(e.g. the location of the result folder) are hardcoded inside the script.
An example usage is:
```
./streamline_gluino.sh "--jets 8" "alpaca_8j_hydra_3layers_UDS_1400_newtag_noisbpred_htscale"
```

This script performs two main steps, achieved by calling two python scripts: [csv_to_root.py](./scripts/csv_to_root.py) and
[plot_gluino.py](./scripts/plot_gluino.py). 




