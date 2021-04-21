import ROOT
import os
import argparse
import errno

# goal: draw different variables for the same or different selections, same binning, same tree 

class myHisto:
    def __init__(self, input_file, tname, sel, var, binning, label, color, xaxis='', overflow=True):
        self.input_file=input_file
        self.tname=tname
        self.var=var
        self.binning=binning
        self.label=label
        self.color=color
        self.xaxis=xaxis
        self.sel = sel
        self.overflow = overflow
    def get_histo(self):
        infile=ROOT.TFile.Open(self.input_file,'READ')
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

def make_plot(input_file, tname, histos_sel, variables, bins, labels, colors, mass, name_selection, all_same_sel=False, unit_area=False, overflow=True, output_folder='./'):
    myHistos = [myHisto(input_file, tname, histos_sel[i], variables[i], bins, labels[i], colors[i], mass+' [GeV]', overflow=overflow) for i in range(len(variables))]
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
    c.SaveAs(output_folder+'/'+name_can+'.pdf')


def options():
    parser = argparse.ArgumentParser(usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-file','-i', required=True, help="Input ROOT file")
    parser.add_argument('--output-folder','-o', required=True, help="Output folder")
    return parser.parse_args()

def main():

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    args = options()

    # create output dir if it doesn't exist
    try:
        os.makedirs(args.output_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        
    selections = ['weight<0.01']
    name_selections = ['all_events']

    colors = [856, 801, 410, 629, 879, 607, 602, 921, 622, 588]
    weight = 'weight'
    bins = (50,0,2500)
    tname = 'tree'

    for mass in ['mt1', 'mt2']:
        variables = [mass+'_reco', mass+'_random']
        labels = ['alpaca','random']
        for isel,sel in enumerate(selections):
            print(sel)
            histos_sel = [sel for i in range(len(variables))]
            for overflow in [True,False]: 
                for unit_area in [True,False]:
                    make_plot(args.input_file, tname, histos_sel, variables, bins, labels, colors, mass, name_selections[isel], all_same_sel=True, unit_area=unit_area, overflow=overflow, output_folder=args.output_folder)


if __name__=='__main__':
    main()


