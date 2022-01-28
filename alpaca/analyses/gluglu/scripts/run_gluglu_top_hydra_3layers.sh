njets=8
zero_jets=2
eos_h5='/eos/user/c/crizzi/RPV/alpaca/h5/mio/ttbar/'
tag='alpaca_'${njets}'j_hydra_3layers_top'
spectators=' --spectators n_jets --spectators event_number --spectators weight --spectators has_good_match --spectators jet_isBTag --spectators jet_isSig'
extra=''
#extra='--extra-jet-fields jet_isSig --extra-jet-fields jet_isGluon'

#alpaca gluglu --input-file ${eos_h5}/top_goodmatch_train.h5 --test-sample 10000 -w  --tag ${tag} --output-dir /eos/user/c/crizzi/RPV/alpaca/results/ --hydra --fflayer 200 200 200 --zero-jets ${zero_jets} --jets ${njets} -t ${spectators} ${extra}

# evaluate on train sample
echo "chiara: evaluate on train sample, truth"
alpaca gluglu --input-file ${eos_h5}/top_goodmatch_train.h5 --test-sample -1 -w --label-roc train_truth --tag ${tag}  --output-dir /eos/user/c/crizzi/RPV/alpaca/results/ --zero-jets ${zero_jets} --jets ${njets} ${spectators} ${extra}
echo "chiara: evaluate on train sample, NO truth"
alpaca gluglu --input-file ${eos_h5}top_train.h5 --test-sample -1 -w --label-roc train_noTruth --tag ${tag} --no-truth  --output-dir /eos/user/c/crizzi/RPV/alpaca/results/ --hydra --fflayer 200 200 200 --zero-jets ${zero_jets} --jets ${njets}  ${spectators} ${extra}  --spectators partonindex
# evaluate on test sample
echo "chiara: evaluate on test sample, truth"
alpaca gluglu --input-file ${eos_h5}/top_goodmatch_test.h5 --test-sample -1 -w --label-roc test_truth --tag ${tag}  --output-dir /eos/user/c/crizzi/RPV/alpaca/results/ --zero-jets ${zero_jets} --jets ${njets} ${spectators} ${extra}
echo "chiara: evaluate on test sample, NO truth"
alpaca gluglu --input-file ${eos_h5}top_test.h5 --test-sample -1 -w --label-roc test_noTruth --tag ${tag} --no-truth  --output-dir /eos/user/c/crizzi/RPV/alpaca/results/ --hydra --fflayer 200 200 200 --zero-jets ${zero_jets} --jets ${njets}  ${spectators} ${extra}  --spectators partonindex

# evaluate on dijet
#alpaca gluglu --input-file ${eos_h5}dijet_MBJ_small.h5 --test-sample -1 -w --label-roc dijet_MBJ --tag ${tag} --spectators event_number --output-dir /eos/user/c/crizzi/RPV/alpaca/results/ --hydra --spectators channel_number --spectators weight_mc --spectators weight_lumi --jets ${njets} --zero-jets ${zero_jets} --no-truth

