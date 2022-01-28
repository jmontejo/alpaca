#!/bin/bash

csv_to_root_options=$1
tag=$2

alpaca_dir="/eos/user/c/crizzi/RPV/alpaca/results"
folder=${alpaca_dir}/${tag}
for sample in 'test' '900' '2400' #'train' #'test' #"test_mg2400_mymatch" "test_mg900_mymatch"
do 
    #echo ${sample}
    cmd="python3 csv_to_root.py --input-alpaca-truth ${folder}/NNoutput_${sample}_truth.csv --input-alpaca-no-truth ${folder}/NNoutput_${sample}_noTruth.csv --merged-df ${folder}/merged_csv_${sample}.csv --merged-df-root ${folder}/merged_csv_${sample}.root --pt-order --output ${folder}/outputtree_${sample}.root ${csv_to_root_options} --build-df  --weights weight --no-input-root" # --presel"
    if [ "${sample}" = "train" ]; then
	cmd=${cmd}" --train-sample"
    fi
    if [ "${sample}" = "test_dijet" ]; then
	cmd=${cmd}" --no-truth"
    fi
    echo ${cmd}
    ${cmd}

    cmd="python3 plot_gluino.py -i ${folder}/outputtree_${sample}.root -o ${folder}/plots_${sample}"
    echo ${cmd}
    ${cmd}

 done

cmd="python plot_overtraining.py --file-train ${folder}/outputtree_train.root --file-test ${folder}/outputtree_test.root -o ."
#echo ${cmd}
#${cmd}

cmd="python3 csv_to_root.py --build-df --merged-df ${folder}/merged_csv_dijet_MBJ.csv --merged-df-root ${folder}/merged_csv_dijet_MBJ.root --input-alpaca-no-truth ${folder}/NNoutput_dijet_MBJ.csv --pt-order --output ${folder}/outputtree_dijet_MBJ.root --no-input-root ${csv_to_root_options} --no-truth --weights weight_lumi weight_mc"
#echo ${cmd}
#${cmd}

cmd="python3 plot_top_MBJ.py -i ${folder}/outputtree_dijet_MBJ.root -o ${folder}/plots_dijet_MBJ"
#echo ${cmd}
#${cmd}
