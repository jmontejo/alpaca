import ROOT

# goal: draw different variables for the same or different selections, same binning, same tree 

mass='mt2'

#out_folder='plots_nopTordered'
#fname = '/eos/user/c/crizzi/RPV/alpaca/root/outtree_nopTchoice.root'
# out_folder='plots_nopTordered_hydra'
out_folder='plots_nopTordered_hydra'
# fname = '/eos/user/c/crizzi/RPV/alpaca/root/outtree_ttbar_v3.root'
# fname = '/afs/cern.ch/user/c/crizzi/storage/RPV/alpaca_mio/scripts/outtree_pTordered_hydra.root'
fname = '/eos/user/c/crizzi/RPV/alpaca/root/outtree_nopTordered_hydra_test_NEW.root'

# variables for all variables plotted with the same selection
selections = ['1','has_truth','pass_chi2_bfixed', 'pass_chi2_bfixed && has_truth', 'pass_anatop']
selections = [s+'  && njets==6' for s in selections]
selections = selections + [s.replace('njets==6','njets>=6') for s in selections]
name_selections = ['all_events', 'has_truth', 'pass_chi2', 'has_truth_and_pass_chi2','pass_anatop']
name_selections = [n+'_exactly_6_jets' for n in name_selections]
name_selections = name_selections + [n.replace('exactly','at_least') for n in name_selections]
print(selections)
print(name_selections)

#variables = [mass+'_chi2_nobfixed',mass+'_true']#,mass+'_chi2_nobfixed',mass+'_true']
variables = [mass+'_chi2_nobfixed', mass+'_chi2_bfixed', mass+'_reco', mass+'_random']
labels = ['chi2-no-bfixed','chi2-bfixed','alpaca','random']

common_sel_u = 'njets>=6 && njets_25>5'
selections_u_tmp = ['pass_chi2_nobfixed', 'pass_chi2_bfixed',  '1',                 'has_truth']#,                    'has_truth']
variables_u =  [mass+'_chi2_nobfixed',    mass+'_chi2_bfixed', mass+'_chi2_nobfixed',mass+'_reco']#, mass+'_true']
labels_u =     ['chi2<10 nobfixed',       'chi2<10 bfixed',   'chi2 all','alpaca has truth']#,            'true has truth']
name_sel_u = 'at_least_6_jets_chi2'
selections_u = [s+'  && '+common_sel_u for s in selections_u_tmp]


# variables for Riccardo's plot
common_sel_r = 'njets==6 && njets_25>5'
selections_r_tmp = ['pass_chi2_nobfixed',                  'pass_chi2_bfixed',              'has_truth',       '1',         '1']#,                    'has_truth']
variables_r =  [mass+'_chi2_nobfixed',                  mass+'_chi2_bfixed',            mass+'_reco',      mass+'_reco', mass+'_chi2_nobfixed']#, mass+'_true']
labels_r =     ['chi2<10 nobfixed',                     'chi2<10 bfixed',               'alpaca has truth','alpaca all', 'chi2 all']#,            'true has truth']
name_sel_r = 'exactly_6_jets'
selections_r = [s+'  && '+common_sel_r for s in selections_r_tmp]

# variables for Riccardo's plot
common_sel_s = 'njets>=6 && njets<=7 && njets_25>5'
name_sel_s = '6_or_7_jets'
selections_s = [s+'  && '+common_sel_s for s in selections_r_tmp]

common_sel_t = 'njets>=6 && njets_25>5'
name_sel_t = 'at_least_6_jets'
selections_t = [s+'  && '+common_sel_t for s in selections_r_tmp]


# kgreen-6
# colors = [410, 856, 607, 801, 629, 879, 602, 921, 622, 588]
colors = [856, 801, 410, 629, 879, 607, 602, 921, 622, 588]
weight = 'weight'
bins = (100,0,500)
tname = 'tree'


# test: look at ttbar input file 
# variables for Riccardo's plot
'''
selections_r = ['pass_chi2',           'pass_chi2',           'has_truth',       '1',         '1']#,                    'has_truth']
variables_r =  [mass+'_chi2_nobfixed', mass+'_chi2_nobfixed', mass+'_reco',      mass+'_reco', mass+'_chi2_nobfixed']#, mass+'_true']
labels_r =     ['chi2<10 nobfixed',  ' chi2<10 bfixed',       'alpaca has truth','alpaca all', 'chi2 all']#,            'true has truth']
name_sel_r = 'different_sel'
fname = ''
tname = ''
'''
#fname, tname, sel, var, bin, label, color

ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

class myHisto:
    def __init__(self, fname, tname, sel, var, binning, label, color, xaxis='', overflow=True):
        self.fname=fname
        self.tname=tname
        self.var=var
        self.binning=binning
        self.label=label
        self.color=color
        self.xaxis=xaxis
        self.sel = sel
        self.overflow = overflow
    def get_histo(self):
        infile=ROOT.TFile.Open(self.fname,'READ')
        t=infile.Get(self.tname)
        h_name = self.var
        h = ROOT.TH1D(h_name, self.label, self.binning[0], self.binning[1], self.binning[2])        
        string_draw=self.var+'>>'+h_name
        t.Draw(string_draw,self.sel,'goff')
        h.GetXaxis().SetTitle(self.xaxis)
        h.Sumw2()
        h.SetLineColor(self.color)
        h.SetLineWidth(2)
        h.SetDirectory(0)
        # add overflow and underflow
        if self.overflow:
            # overflow
            h_of = ROOT.TH1D(h_name+'_of', self.label+' overflow', self.binning[0], self.binning[1], self.binning[2])
            string_draw_of = str(self.binning[2]) + " - (0.5*("+ str(self.binning[2])+"-"+str(self.binning[1])+")/"+str(self.binning[0]) +") >>"+h_name+"_of"
            sel_of = '(('+self.sel+') && ('+self.var+'>'+str(self.binning[2])+'))'
            # print(string_draw_of)
            # print(sel_of)
            t.Draw(string_draw_of,sel_of,'goff')
            h.Add(h_of)
            # underflow
            h_uf = ROOT.TH1D(h_name+'_uf', self.label+' underflow', self.binning[0], self.binning[1], self.binning[2])
            string_draw_uf = str(self.binning[1]) + " + (0.5*("+ str(self.binning[2])+"-"+str(self.binning[1])+")/"+str(self.binning[0]) +") >>"+h_name+"_uf"
            sel_uf = '(('+self.sel+') && ('+self.var+'<'+str(self.binning[1])+'))'
            # print(string_draw_uf)
            # print(sel_uf)
            t.Draw(string_draw_uf,sel_uf,'goff')
            h.Add(h_uf)
            
        # done adding overflow and underflow
        infile.Close()
        return h


def write_text(write, pad):
    pad.cd()
    text =  ROOT.TLatex()
    text.SetNDC()
    text.SetTextAlign( 11 )
    text.SetTextFont( 42 )
    text.SetTextSize( 0.045 )
    text.SetTextColor( 1 )
    y = 0.86
    for t in write:
        text.DrawLatex(0.15,y, t)
        y = y-0.046

def make_plot(fname, tname, histos_sel, variables, bins, labels, colors, mass, name_selection, all_same_sel=False, unit_area=False, overflow=True):
    myHistos = [myHisto(fname, tname, histos_sel[i], variables[i], bins, labels[i], colors[i], mass+' [GeV]', overflow=overflow) for i in range(len(variables))]
    histos = [h.get_histo() for h in myHistos]
    if unit_area:
        for h in histos:
            h.Scale(1./h.Integral())
            #h.Scale(1./5.)
    # print(histos)
    #for h in histos: print(h.GetEntries(), h.GetMean(), h.Integral())
    c = ROOT.TCanvas("can"+name_selection,"can"+name_selection,750,600)
    pad1 = ROOT.TPad("pad1", "pad1",0.0,0.0,1.0,1.0,21)
    pad1.SetFillStyle(0)
    pad1.SetFillColor(0)
    pad1.SetTopMargin(0.08)
    pad1.SetBottomMargin(0.13)
    pad1.SetTickx()
    pad1.SetTicky()
    pad1.Draw()
    pad1.cd()
    hist_max = max([h.GetMaximum() for h in histos])
    for ih,h in enumerate(histos): 
        h.SetMaximum(1.2*hist_max)
        h.Draw() if ih==0 else h.Draw('same')
        h.Draw("hist same")
    #legend
    leg=ROOT.TLegend(0.6, 0.6, 0.875, 0.87)
    leg.SetFillStyle(0)
    leg.SetLineColor(0)
    leg.SetLineWidth(0)
    for h in histos: leg.AddEntry(h, h.GetTitle(), 'l')
    leg.Draw()
    write= ['ATLAS Internal','Sel: '+name_selection.replace('_',' ')]
    if all_same_sel:
        write.append('Events: '+str(histos[0].GetEntries()))
    write_text(write, pad1)

    name_can = mass+'_'+name_selection
    if unit_area: name_can += '_unit_area'
    if overflow: name_can += '_overflow'
    c.SaveAs(out_folder+'/'+name_can+'.pdf')

'''
for overflow in [True,False]: 
    for unit_area in [True,False]:
        make_plot(fname, tname, selections_u, variables_u, bins, labels_u, colors, mass, name_sel_u, all_same_sel=False, unit_area=unit_area, overflow=overflow)
'''

for overflow in [True,False]: 
    for unit_area in [True,False]:
        make_plot(fname, tname, selections_r, variables_r, bins, labels_r, colors, mass, name_sel_r, all_same_sel=False, unit_area=unit_area, overflow=overflow)
        make_plot(fname, tname, selections_s, variables_r, bins, labels_r, colors, mass, name_sel_s, all_same_sel=False, unit_area=unit_area, overflow=overflow)
        make_plot(fname, tname, selections_t, variables_r, bins, labels_r, colors, mass, name_sel_t, all_same_sel=False, unit_area=unit_area, overflow=overflow)

for isel,sel in enumerate(selections):
    print(sel)
    histos_sel = [sel for i in range(len(variables))]
    for overflow in [True,False]: 
        for unit_area in [True,False]:
            make_plot(fname, tname, histos_sel, variables, bins, labels, colors, mass, name_selections[isel], all_same_sel=True, unit_area=unit_area, overflow=overflow)
            make_plot(fname, tname, histos_sel, variables, bins, labels, colors, mass, name_selections[isel], all_same_sel=True, unit_area=unit_area, overflow=overflow)




