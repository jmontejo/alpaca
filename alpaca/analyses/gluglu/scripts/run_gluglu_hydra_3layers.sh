njets=8
zero_jets=2
eos_h5='/eos/user/c/crizzi/RPV/alpaca/h5/mio/FT_022621/'
eos_h5='../'
outdir='delme/'
tag='alpaca_'${njets}'j_hydra_3layers_UDS_1400_partonindex'
spectators=' --spectators n_jets --spectators event_number --spectators weight --spectators has_good_match --spectators jet_isGluon --spectators jet_isSig'
extra=''
#extra='--extra-jet-fields jet_isSig --extra-jet-fields jet_isGluon'

alpaca gluglu --input-file ${eos_h5}truthmatched_gluino_truth_UDS_1400_mymatch_train.h5 --test-sample 1000 -w  --tag ${tag} --output-dir $outdir --hydra --fflayer 200 200 200 --zero-jets ${zero_jets} --jets ${njets} -t ${spectators} ${extra}

# evaluate on train sample
alpaca gluglu --input-file ${eos_h5}truthmatched_gluino_truth_UDS_1400_mymatch_train.h5 --test-sample -1 -w --label-roc train_truth --tag ${tag}  --output-dir $outdir --zero-jets ${zero_jets} --jets ${njets} ${spectators} ${extra}
alpaca gluglu --input-file ${eos_h5}truthmatched_gluino_noTruth_UDS_1400_mymatch_train.h5 --test-sample -1 -w --label-roc train_noTruth --tag ${tag} --no-truth  --output-dir $outdir --zero-jets ${zero_jets} --jets ${njets} ${spectators} ${extra}  --spectators partonindex
# evaluate on test sample
alpaca gluglu --input-file ${eos_h5}truthmatched_gluino_truth_UDS_1400_mymatch_test.h5 --test-sample -1 -w --label-roc test_truth --tag ${tag}  --output-dir $outdir --zero-jets ${zero_jets} --jets ${njets} ${spectators} ${extra}
alpaca gluglu --input-file ${eos_h5}truthmatched_gluino_noTruth_UDS_1400_mymatch_test.h5 --test-sample -1 -w --label-roc test_noTruth --tag ${tag} --no-truth  --output-dir $outdir --hydra --fflayer 200 200 200 --zero-jets ${zero_jets} --jets ${njets}  ${spectators} ${extra}  --spectators partonindex

# evaluate on dijet
#alpaca gluglu --input-file ${eos_h5}dijet_MBJ_small.h5 --test-sample -1 -w --label-roc dijet_MBJ --tag ${tag} --spectators event_number --output-dir $outdir --hydra --spectators channel_number --spectators weight_mc --spectators weight_lumi --jets ${njets} --zero-jets ${zero_jets} --no-truth

