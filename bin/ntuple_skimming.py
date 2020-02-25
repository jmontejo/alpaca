#!/usr/bin/env python
from pathlib2 import Path

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)



if __name__ == '__main__':
   p_samples = Path('/home/users/p/poggi/scratch/xsttbar_samples/')
   p_data = p_samples.glob('user.rpoggi.period*.physics_Main.DAOD_SUSY4*.TTDIFFXS34_R21_allhad_resolved.root/*.root')
   #p_data = p_samples.glob('user.rpoggi.410471*e5984*XS34*/*.root')

   p_out = p_samples / 'multijet' / 'data_a0b0'
   p_out.mkdir(parents=True, exist_ok=True)
   for p in p_data:
       print('Reading from file: {}'.format(p))
       orig_f = ROOT.TFile.Open(str(p), 'READ')
       orig_nominal = orig_f.Get('nominal')
       target_f = ROOT.TFile.Open(str(p_out / p.name), 'RECREATE')
       target_nominal = orig_nominal.CloneTree(0)

       for event in orig_nominal:
           if (event.jet_n == 6 and event.reco_Chi2Fitted < 10. and
               event.reco_DRbWMax < 2.2 and event.reco_DRbb > 2.0 and
               #(event.reco_t1_m < 120000 or event.reco_t1_m > 250000 or
               # event.reco_t2_m < 120000 or event.reco_t2_m > 250000) and
               event.reco_bjets == 0):
            
          #if (event.jet_n == 6 and event.reco_bjets == 2):
               target_nominal.Fill()
       target_nominal.Print()
       target_nominal.AutoSave()

       orig_f.Close()
       target_f.Close()
