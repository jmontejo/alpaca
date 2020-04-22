ALPACA
======

ML project for top-like tagger.

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
