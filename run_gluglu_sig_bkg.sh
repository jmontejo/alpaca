njets=12
zero_jets=6
eos_h5='/eos/user/c/crizzi/RPV/alpaca/h5/mio/'
tag='alpaca_sigbkg_'${njets}'j_colalola_3layers_UDBUDS_1400'
out_dir='/eos/user/c/crizzi/RPV/alpaca/results/'

alpaca gluglu -i ${eos_h5}truthmatched_gluino_UDBUDS_1400_mymatch_sel.h5 -ic 1 -i ${eos_h5}truthmatched_gluino_UDS_1400_mymatch_sel_train.h5 -ic 1 -i ${eos_h5}/FT_dijet_mymatch_sel_train.h5 -ic 0 --test-sample 1000 -w --spectators event_number --spectators n_jets --tag ${tag} --output-dir ${out_dir} --cola-lola --fflayer 200 200 200 --zero-jets ${zero_jets} --jets ${njets} -t --shuffle-events --outputs 1 --categories 1 --spectators pass_sel --spectators weight 

# evaluate on test sample
alpaca gluglu -i ${eos_h5}truthmatched_gluino_UDS_1400_mymatch_sel_test.h5 -ic 1 -i ${eos_h5}/FT_dijet_mymatch_sel_test.h5 -ic 0 --test-sample -1 -w --label-roc test --tag ${tag} --spectators event_number --spectators n_jets --output-dir ${out_dir} --zero-jets ${zero_jets} --jets ${njets} --outputs 1 --categories 1 --spectators pass_sel --spectators weight
alpaca gluglu -i ${eos_h5}truthmatched_gluino_UDS_1400_mymatch_sel_test.h5 -ic 1 --test-sample -1 -w --label-roc test_signal --tag ${tag} --spectators event_number --spectators n_jets --output-dir ${out_dir} --zero-jets ${zero_jets} --jets ${njets} --outputs 1 --categories 1 --spectators pass_sel --spectators weight
alpaca gluglu -i ${eos_h5}/FT_dijet_mymatch_sel_test.h5 -ic 0 --test-sample -1 -w --label-roc test_dijet --tag ${tag} --spectators event_number --spectators n_jets --output-dir ${out_dir} --zero-jets ${zero_jets} --jets ${njets} --outputs 1 --categories 1 --spectators pass_sel --spectators weight 
alpaca gluglu -i ${eos_h5}truthmatched_gluino_UDS_1400_mymatch_sel_train.h5 -ic 1 -i ${eos_h5}/FT_dijet_mymatch_sel_train.h5 -ic 0 --test-sample -1 -w --label-roc train --tag ${tag} --spectators event_number --spectators n_jets --output-dir ${out_dir} --zero-jets ${zero_jets} --jets ${njets} --outputs 1 --categories 1 --spectators pass_sel --spectators weight
#alpaca gluglu --input-file ${eos_h5}truthmatched_gluino_noTruth_UDS_1400_train.h5 --test-sample -1 -w --label-roc train_noTruth --tag ${tag} --no-truth --spectators event_number --spectators n_jets --output-dir ${out_dir} --zero-jets ${zero_jets} --jets ${njets} --spectators partonindex --spectators good_match
# evaluate on test sample
#alpaca gluglu --input-file ${eos_h5}truthmatched_gluino_truth_UDS_1400_test.h5 --test-sample -1 -w --label-roc test_truth --tag ${tag} --spectators event_number --spectators good_match --spectators n_jets --output-dir ${out_dir} --zero-jets ${zero_jets} --jets ${njets}
#alpaca gluglu --input-file ${eos_h5}truthmatched_gluino_noTruth_UDS_1400_test.h5 --test-sample -1 -w --label-roc test_noTruth --tag ${tag} --no-truth --spectators event_number --spectators n_jets --output-dir ${out_dir} --cola-lola --fflayer 200 200 200 --zero-jets ${zero_jets} --jets ${njets} --spectators partonindex --spectators good_match 

# evaluate on MBJ ttbar sample non all had 

# evaluate on dijet
#alpaca gluglu --input-file ${eos_h5}dijet_MBJ_small.h5 --test-sample -1 -w --label-roc dijet_MBJ --tag ${tag} --spectators event_number --output-dir ${out_dir} --cola-lola --spectators channel_number --spectators weight_mc --spectators weight_lumi --jets ${njets} --zero-jets ${zero_jets} --no-truth

