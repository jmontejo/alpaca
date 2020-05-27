# Jet-parton matching with uproot/pandas

import uproot, uproot_methods, math
import numpy as np

# Define parton labels and corresponding indices
# 0 denotes a failure to match
partons = [ '{}_from_{}'.format(child,parent) for parent in ['t','tbar'] for child in ['b','Wdecay1','Wdecay2']]
partonindex = {parton:iparton for iparton,parton in enumerate([None]+partons)}

# Open root file
fname = "test_in.410471.root"
print("Working on file '{}'".format(fname))
# Access trees as dataframe iterators reading multiple input files
print("Making pandas")
nom_iter = uproot.pandas.iterate(fname,"nominal",entrysteps=float("inf"),branches=["eventNumber","jet_*"])
truth_iter = uproot.pandas.iterate(fname,"truth",entrysteps=float("inf"),branches=["MC_b_from_t*","MC_Wdecay*_from_t*"])
# Set the truth df index so we can use "join"
# Join function seems to preserve subentry, which merge does not
truth_eventN_iter = uproot.pandas.iterate(fname,"truth",entrysteps=float("inf"),branches="eventNumber")

ifile = 0
for nom_df, truth_df, truth_eventN_df in zip(nom_iter,truth_iter,truth_eventN_iter):
    print("Processing file {}".format(ifile))
    # Merge dataframes on "eventNumber" key
    # Merging truth into nominal retains only entries in nominal tree
    truth_df.index = truth_eventN_df["eventNumber"]
    joined_df = nom_df.join(truth_df,on="eventNumber")
    print("  Length of joined df: {}".format(len(joined_df)))

    print("  Computing dR")
    # Compute delta eta and delta phi
    twopi = 2*math.pi # mod takes precedence over times, so precompute
    for parton in partons:
        joined_df["jet_deta_{}".format(parton)] = joined_df["jet_eta"] - joined_df["MC_{}_eta".format(parton)]
        joined_df["jet_dphi_{}".format(parton)] = (joined_df["jet_phi"] - joined_df["MC_{}_phi".format(parton)] + math.pi) % twopi - math.pi
        joined_df["jet_dR_{}".format(parton)] = np.sqrt(joined_df["jet_deta_{}".format(parton)]**2 + joined_df["jet_dphi_{}".format(parton)]**2)

    # Apply jet-parton matching
    print("  Matching partons to jets")
    dRbranches = ["jet_dR_{}".format(parton) for parton in partons]
    match_dRcut = 0.4
    joined_df["partonlabel"] = np.where(joined_df[dRbranches].min(axis=1)<match_dRcut,joined_df[dRbranches].idxmin(axis=1).str.lstrip("jet_dR"),None)
    joined_df["partonindex"] = joined_df["partonlabel"].map(partonindex)

    # Define columns in cartesian coordinates, needed later for the network
    joined_df["jet_px"] = joined_df["jet_pt"]*np.cos(joined_df["jet_phi"])
    joined_df["jet_py"] = joined_df["jet_pt"]*np.sin(joined_df["jet_phi"])
    joined_df["jet_pz"] = joined_df["jet_pt"]*np.sinh(joined_df["jet_eta"])

    # Truncate the jet vector
    maxjets = 10
    truncated_df = joined_df.drop(range(maxjets,50),level="subentry")

    # Remove cases where a truth parton is missing (how?).
    partonsexist = np.min(truncated_df[["MC_{}_m".format(parton) for parton in partons]],axis=1)!=0
    sixmatches = truncated_df["partonlabel"].count(level="entry")==6
    filtercond = partonsexist & sixmatches

    filtered_df = truncated_df[filtercond]

    # Extract just jets with partons matched below the chosen deltaR cut
    goodmatch = filtered_df[dRbranches].min(axis=1) < match_dRcut
    matched_jets = filtered_df[goodmatch]

    # Reshape the df to have one event per row,
    # filling each overflow entry with 0's
    # Extract just the cartesian coordinates and the parton index
    jet_cartesian = ["jet_"+comp for comp in ["px","py","pz","e"]]
    unstacked_df = filtered_df[jet_cartesian+["partonindex"]].unstack(fill_value=0)

    # Write out to an hdf5 file
    unstacked_df.to_hdf("truthmatched.h5",key="df",mode='w' if ifile==0 else 'a',append=True)

    ifile+=1
