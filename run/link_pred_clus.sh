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
AUG_FLAG=${AUG_FLAG:-0}
SEG_FLAG=${SEG_FLAG:-0}
USE_VAL_FLAG=${USE_VAL_FLAG:-0}
REV_DUP_FLAG=${REV_DUP_FLAG:-0}
TRAIN_GUI_FLAG=${TRAIN_GUI_FLAG:-0}
# --------------------

launch () {
    full_settings=($@)
    cfg_name=${full_settings[0]}
    # ckpt_name=${full_settings[1]}
    seed=${full_settings[1]}
    thres=${full_settings[2]}
    settings=${full_settings[@]:3}

    extra_config=""
    extra_args="--thres ${thres} "

    if [[ $USE_VAL_FLAG == 1 ]]; then
        extra_args+="--use_val_edges t "
    fi

    if [[ $REV_DUP_FLAG == 1 ]]; then
        extra_args+="--remove_dup t "
    fi

    if [[ $TRAIN_GUI_FLAG == 1 ]]; then
        extra_args+="--train_guidance t "
    fi

    if [[ $AUG_FLAG == 1 ]]; then
        extra_args+="--augment t "

        if [[ $SEG_FLAG == 1 ]]; then
            extra_config+="augment.segment_flag=True "
        else
            extra_config+="augment.segment_flag=False "
        fi

    fi

    config_file="config/link_pred_clus/${cfg_name}.yaml"

    script="python link_pred_clus.py --config ${config_file} --seed ${seed} ${extra_args} ${settings[@]} ${extra_config}"

    echo ${script} && [[ $TEST_FLAG == 0 ]] && eval ${script}
}


# AUG_FLAG=0 USE_VAL_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/link_pred_clus.sh
# AUG_FLAG=1 SEG_FLAG=1 USE_VAL_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/link_pred_clus.sh
# AUG_FLAG=1 SEG_FLAG=1 TRAIN_GUI_FLAG=1 USE_VAL_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/link_pred_clus.sh

# AUG_FLAG=0 CUDA_VISIBLE_DEVICES=0 bash run/link_pred_clus.sh
# AUG_FLAG=1 SEG_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/link_pred_clus.sh
# AUG_FLAG=1 SEG_FLAG=1 TRAIN_GUI_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/link_pred_clus.sh

# thres=None

# thres=0.5
# thres=0.6
# thres=0.7
# thres=0.8

# thres=0.9
# thres=0.99
thres=0.999

# thres=0.9999
# thres=0.99999
# thres=0.999999

# thres=0.9999999
# thres=0.99999999
# thres=0.999999999

# thres=0.9999999999
# thres=0.99999999999
# thres=0.999999999999


echo thres=${thres}

seeds=(0)

seeds=(0 1 2 3 4 5 6 7 8 9)

# seeds=(0 1 2 3 4)
# seeds=(5 6 7 8 9)

# seeds=(0 1 2 3)
# seeds=(4 5 6 7)
# seeds=(8 9)

# seeds=(0 1 2)
# seeds=(3 4 5)
# seeds=(6 7)
# seeds=(8 9)

# seeds=(0 1)
# seeds=(2 3)
# seeds=(4 5)
# seeds=(6 7)
# seeds=(8 9)


for seed in ${seeds[@]}; do

echo seed=${seed}

## nr
# launch cora_nr ${seed} ${thres}&
# launch citeseer_nr ${seed} ${thres}&
# launch pubmed_nr ${seed} ${thres}&
# launch ogbl-citation2_nr ${seed} ${thres}&
# launch ogbl-ppa_nr ${seed} ${thres}&
# launch ogbl-collab_nr ${seed} ${thres}&
# launch ogbl-ddi_nr ${seed} ${thres}&

## nr_snap
# launch cora_nr_snap ${seed} ${thres}&
# launch citeseer_nr_snap ${seed} ${thres}&
# launch pubmed_nr_snap ${seed} ${thres}&

# launch ogbl-collab_nr_snap ${seed} ${thres}&
# launch flickr_nr_snap ${seed} ${thres}&

# launch ogbl-citation2_nr_snap ${seed} ${thres}&
# launch ogbl-ppa_nr_snap ${seed} ${thres}&
# launch ogbl-ddi_nr_snap ${seed} ${thres}&


## TRAIN_GUI_FLAG=1
# launch ogbl-collab_nr_snap_edge ${seed} ${thres} --prefix edge_cn&
# launch ogbl-collab_nr_snap_node ${seed} ${thres} --prefix node_degree&

# launch flickr_nr_snap_edge ${seed} ${thres} --prefix edge_cn&
# launch flickr_nr_snap_node ${seed} ${thres} --prefix node_degree&

wait

done

# wait





