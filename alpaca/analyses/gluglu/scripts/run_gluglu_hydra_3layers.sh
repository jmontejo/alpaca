njets=8
zero_jets=2 # njets - 6
tag='alpaca_'${njets}'j_hydra_3layers_UDS_1400_newtag_noisbpred_htscale'
spectators=' --spectators n_jets --spectators event_number --spectators weight --spectators has_good_match --spectators jet_isBTag --spectators jet_isSig'
extra='--scale-e'
#extra='--extra-jet-fields jet_isSig --extra-jet-fields jet_isGluon'
#dids_train='504518' # UDB 1400
dids_train='504539' # UDS 1400
dids_900='504534' # USD
dids_2400='504549' # UDS

#h5_folder='/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/h5/workshop/mc16e/'
#results_folder='/eos/user/c/crizzi/RPV/alpaca/results/'
h5_folder='/Users/crizzi/lavoro/SUSY/MJ_RPV/alpaca_ultimo/input/mc16e/'
results_folder='/Users/crizzi/lavoro/SUSY/MJ_RPV/alpaca_ultimo/results/'

# training 
alpaca gluglu --input-file ${h5_folder}${dids_train}_goodmatch_train.h5 --test-sample 1000 -w  --tag ${tag} --output-dir ${results_folder} --hydra --fflayer 200 200 200 --zero-jets ${zero_jets} --jets ${njets} -t ${spectators} ${extra}

# evaluate on train sample
alpaca gluglu --input-file ${h5_folder}${dids_train}_goodmatch_train.h5 --test-sample -1 -w --label-roc train_truth --tag ${tag}  --output-dir ${results_folder} --zero-jets ${zero_jets} --jets ${njets} ${spectators} ${extra}
alpaca gluglu --input-file ${h5_folder}${dids_train}_train.h5 --test-sample -1 -w --label-roc train_noTruth --tag ${tag} --no-truth  --output-dir ${results_folder} --zero-jets ${zero_jets} --jets ${njets} ${spectators} ${extra}  --spectators partonindex
# evaluate on test sample
alpaca gluglu --input-file ${h5_folder}${dids_train}_goodmatch_test.h5 --test-sample -1 -w --label-roc test_truth --tag ${tag}  --output-dir ${results_folder} --zero-jets ${zero_jets} --jets ${njets} ${spectators} ${extra}
alpaca gluglu --input-file ${h5_folder}${dids_train}_test.h5 --test-sample -1 -w --label-roc test_noTruth --tag ${tag} --no-truth  --output-dir ${results_folder} --hydra --fflayer 200 200 200 --zero-jets ${zero_jets} --jets ${njets}  ${spectators} ${extra}  --spectators partonindex
# evaluate on 900
alpaca gluglu --input-file ${h5_folder}${dids_900}_goodmatch.h5 --test-sample -1 -w --label-roc 900_truth --tag ${tag}  --output-dir ${results_folder} --zero-jets ${zero_jets} --jets ${njets} ${spectators} ${extra}
alpaca gluglu --input-file ${h5_folder}${dids_900}.h5 --test-sample -1 -w --label-roc 900_noTruth --tag ${tag} --no-truth  --output-dir ${results_folder} --zero-jets ${zero_jets} --jets ${njets} ${spectators} ${extra}  --spectators partonindex
# evaluate on 2400
alpaca gluglu --input-file ${h5_folder}${dids_2400}_goodmatch.h5 --test-sample -1 -w --label-roc 2400_truth --tag ${tag}  --output-dir ${results_folder} --zero-jets ${zero_jets} --jets ${njets} ${spectators} ${extra}
alpaca gluglu --input-file ${h5_folder}${dids_2400}.h5 --test-sample -1 -w --label-roc 2400_noTruth --tag ${tag} --no-truth  --output-dir ${results_folder} --zero-jets ${zero_jets} --jets ${njets} ${spectators} ${extra}  --spectators partonindex

# evaluate on dijet
#echo "alpaca gluglu --input-file ${h5_folder}qcd_cleaned_test.h5 --test-sample 100 -w --label-roc qcd --tag ${tag} --no-truth  --output-dir ${results_folder} --hydra --fflayer 200 200 200 --zero-jets ${zero_jets} --jets ${njets}  ${spectators} ${extra} --spectators partonindex"
alpaca gluglu --input-file ${h5_folder}qcd_cleaned_test.h5 --test-sample 100 -w --label-roc qcd --tag ${tag} --no-truth  --output-dir ${results_folder} --hydra --fflayer 200 200 200 --zero-jets ${zero_jets} --jets ${njets}  ${spectators} ${extra} --spectators partonindex

