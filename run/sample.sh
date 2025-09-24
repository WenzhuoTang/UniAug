#!/bin/bash --login
# Uncomment specific block of interest and run
# $ bash script.sh
#
# Specify CUDA devices
# $ CUDA_VISIBLE_DEVICES=1 bash script.sh
#
# To view the generated script without executing, pass the TEST envar as 1
# $ TEST=1 bash script.sh

trap "echo ERROR && exit 1" ERR

# --------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TEST_FLAG=${TEST_FLAG:-0}
# --------------------

launch () {
    full_settings=($@)
    cfg_name=${full_settings[0]}
    prefix=${full_settings[1]}
    seed=${full_settings[2]}
    settings=${full_settings[@]:3}

    config_file="pretrain_config/binary/${cfg_name}.yaml"

    script="python sample.py --config ${config_file}"
    script+=" --prefix ${prefix} ${settings[@]}"

    echo ${script} && [[ $TEST_FLAG == 0 ]] && eval ${script}
}


# 5 repeats -------------------------------------------------------------------
# seeds=(1 2 3 4 5)
seeds=(1)

for seed in ${seeds[@]}; do

echo seed=${seed}

## 
# launch comm_gnn comm_gnn_gat_l3_deg_emb ${seed} sample.ckpt_path=checkpoints/community/comm_gnn_gat_l3_deg_emb-r.1-Feb13-20:47:05.pth sample.sample_from_empty=False &
# launch comm_gnn comm_gnn_gat_l3_deg_emb_empty ${seed} sample.ckpt_path=checkpoints/community/comm_gnn_gat_l3_deg_emb-r.1-Feb13-20:47:05.pth sample.sample_from_empty=True &

# launch comm_tgnn comm_tgnn_l5 ${seed} sample.ckpt_path=checkpoints/community/comm_tgnn_l5-r.1-Feb13-20:46:09.pth &
# launch comm_tgnn comm_tgnn_l5_empty ${seed} sample.ckpt_path=checkpoints/community/comm_tgnn_l5-r.1-Feb13-20:46:09.pth sample.sample_from_empty=True &

wait

done


