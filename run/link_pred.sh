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
    ckpt_name=${full_settings[1]}
    seed=${full_settings[2]}
    thres=${full_settings[3]}
    settings=${full_settings[@]:4}

    extra_config=""
    extra_args="--thres ${thres} "

    ## modifying
    # extra_config+="augment.sample.inpaint_every_step=True "
    
    # extra_config+="augment.sample.inpaint_every_step=False "

    # extra_args+="--prefix patience-100 "
    # extra_config+="train.patience=100 "

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
        ckpt_path="checkpoints/network_repository/"

        if [[ $SEG_FLAG == 1 ]]; then
            extra_config+="augment.segment_flag=True "
        else
            extra_config+="augment.segment_flag=False "
        fi

        if [[ $ckpt_name == nr_gnn_gat ]]; then
            ckpt_path+="nr_gnn_gat-Feb06-21:31:39.pth"
        elif [[ $ckpt_name == nr_gnn_gcn ]]; then
            ckpt_path+="nr_gnn_gcn-Feb08-19:31:09.pth"
        elif [[ $ckpt_name == nr_gnn_sage ]]; then
            ckpt_path+="nr_gnn_sage-Feb09-02:20:36.pth"
        elif [[ $ckpt_name == nr_tgnn ]]; then
            ckpt_path+="nr_tgnn-Feb08-21:00:07.pth"
        elif [[ $ckpt_name == nr_gnn_gt_l4 ]]; then
            ckpt_path+="nr_gnn_gt_l4-r.1-Feb15-15:08:36.pth"
        elif [[ $ckpt_name == nr_gnn_gt_l4_deg_emb ]]; then
            ckpt_path+="nr_gnn_gt_l4_deg_emb-r.1-Feb21-21:46:19.pth"
        elif [[ $ckpt_name == nr_gnn_gt_l4_deg_lin ]]; then
            ckpt_path+="nr_gnn_gt_l4_deg_lin-r.1-Feb21-21:45:49.pth"
        elif [[ $ckpt_name == nr_snap_gnn_gt_l4_deg_emb ]]; then
            ckpt_path="checkpoints/network_repository_large_network_repository_snap/"
            ckpt_path+="nr_snap_gnn_gt_l4_deg_emb-r.1-Feb22-19:43:08_1000.pth"
        elif [[ $ckpt_name == nr_snap_gnn_gt_l4_deg_lin ]]; then
            ckpt_path="checkpoints/network_repository_large_network_repository_snap/"
            ckpt_path+="nr_snap_gnn_gt_l4_deg_lin-r.1-Feb22-19:42:52_5000.pth"
        else
            echo Unknown ckpt name $ckpt_name && exit 1
        fi
        extra_config+="augment.ckpt_path=${ckpt_path} "

    else
        extra_config=""
    fi

    config_file="config/link_pred/${cfg_name}.yaml"

    script="python link_pred.py --config ${config_file} --seed ${seed} ${extra_args} ${extra_config} ${settings[@]} "

    echo ${script} && [[ $TEST_FLAG == 0 ]] && eval ${script}
}


# AUG_FLAG=0 USE_VAL_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/link_pred.sh
# AUG_FLAG=1 SEG_FLAG=1 USE_VAL_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/link_pred.sh
# AUG_FLAG=1 SEG_FLAG=1 TRAIN_GUI_FLAG=1 USE_VAL_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/link_pred.sh

# AUG_FLAG=0 CUDA_VISIBLE_DEVICES=0 bash run/link_pred.sh
# AUG_FLAG=1 SEG_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/link_pred.sh
# AUG_FLAG=1 SEG_FLAG=1 TRAIN_GUI_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/link_pred.sh

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

# seeds=(0)

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


for seed in ${seeds[@]}; do

echo seed=${seed}

### nr_gnn_gat
# launch cora nr_gnn_gat ${seed} ${thres}&
# launch citeseer nr_gnn_gat ${seed} ${thres}&
# launch pubmed nr_gnn_gat ${seed} ${thres}&

### nr_gnn_gcn
# launch cora nr_gnn_gcn ${seed} ${thres}&
# launch citeseer nr_gnn_gcn ${seed} ${thres}&
# launch pubmed nr_gnn_gcn ${seed} ${thres}&

### nr_gnn_sage
# launch cora nr_gnn_sage ${seed} ${thres}&
# launch citeseer nr_gnn_sage ${seed} ${thres}&
# launch pubmed nr_gnn_sage ${seed} ${thres}&

### nr_tgnn
# launch cora nr_tgnn ${seed} ${thres}&
# launch citeseer nr_tgnn ${seed} ${thres}&
# launch pubmed nr_tgnn ${seed} ${thres}&

### nr_gnn_gt_l4
# launch cora nr_gnn_gt_l4 ${seed} ${thres}&
# launch citeseer nr_gnn_gt_l4 ${seed} ${thres}&
# launch pubmed nr_gnn_gt_l4 ${seed} ${thres}&

### nr_gnn_gt_l4_deg_emb
## no guidance
# launch cora nr_gnn_gt_l4_deg_emb ${seed} ${thres}&
# launch citeseer nr_gnn_gt_l4_deg_emb ${seed} ${thres}&
# launch pubmed nr_gnn_gt_l4_deg_emb ${seed} ${thres}&
# launch ogbl-citation2 nr_gnn_gt_l4_deg_emb ${seed} ${thres}&
# launch ogbl-ppa nr_gnn_gt_l4_deg_emb ${seed} ${thres}&
# launch ogbl-collab nr_gnn_gt_l4_deg_emb ${seed} ${thres}&
# launch ogbl-ddi nr_gnn_gt_l4_deg_emb ${seed} ${thres}&

### nr_gnn_gt_l4_deg_lin
## no guidance
# launch cora nr_gnn_gt_l4_deg_lin ${seed} ${thres}&
# launch citeseer nr_gnn_gt_l4_deg_lin ${seed} ${thres}&
# launch pubmed nr_gnn_gt_l4_deg_lin ${seed} ${thres}&

# launch ogbl-citation2 nr_gnn_gt_l4_deg_lin ${seed} ${thres}&
# launch ogbl-ppa nr_gnn_gt_l4_deg_lin ${seed} ${thres}&
# launch ogbl-ddi nr_gnn_gt_l4_deg_lin ${seed} ${thres}&

# launch ogbl-collab nr_gnn_gt_l4_deg_lin ${seed} ${thres}&  # use_val=True
# launch power nr_gnn_gt_l4_deg_lin ${seed} ${thres}&  # seg=False
# launch yst nr_gnn_gt_l4_deg_lin ${seed} ${thres}&  # seg=False
# launch erd nr_gnn_gt_l4_deg_lin ${seed} ${thres}&
# launch photo nr_gnn_gt_l4_deg_lin ${seed} ${thres}&
# launch flickr nr_gnn_gt_l4_deg_lin ${seed} ${thres}&

### nr_snap_gnn_gt_l4_deg_emb
## no guidance
# launch cora nr_snap_gnn_gt_l4_deg_emb ${seed} ${thres}&
# launch citeseer nr_snap_gnn_gt_l4_deg_emb ${seed} ${thres}&
# launch pubmed nr_snap_gnn_gt_l4_deg_emb ${seed} ${thres}&
# launch ogbl-citation2 nr_snap_gnn_gt_l4_deg_emb ${seed} ${thres}&
# launch ogbl-ppa nr_snap_gnn_gt_l4_deg_emb ${seed} ${thres}&
# launch ogbl-collab nr_snap_gnn_gt_l4_deg_emb ${seed} ${thres}&
# launch ogbl-ddi nr_snap_gnn_gt_l4_deg_emb ${seed} ${thres}&

### nr_snap_gnn_gt_l4_deg_lin
## no guidance
# launch cora nr_snap_gnn_gt_l4_deg_lin ${seed} ${thres}&
# launch citeseer nr_snap_gnn_gt_l4_deg_lin ${seed} ${thres}&
# launch pubmed nr_snap_gnn_gt_l4_deg_lin ${seed} ${thres}&

# launch ogbl-citation2 nr_snap_gnn_gt_l4_deg_lin ${seed} ${thres}&
# launch ogbl-ppa nr_snap_gnn_gt_l4_deg_lin ${seed} ${thres}&
# launch ogbl-ddi nr_snap_gnn_gt_l4_deg_lin ${seed} ${thres}&

# launch ogbl-collab nr_snap_gnn_gt_l4_deg_lin ${seed} ${thres}&  # use_val=True
# launch power nr_snap_gnn_gt_l4_deg_lin ${seed} ${thres}&  # seg=False
# launch yst nr_snap_gnn_gt_l4_deg_lin ${seed} ${thres}&  # seg=False
# launch erd nr_snap_gnn_gt_l4_deg_lin ${seed} ${thres}&
# launch photo nr_snap_gnn_gt_l4_deg_lin ${seed} ${thres}&
# launch flickr nr_snap_gnn_gt_l4_deg_lin ${seed} ${thres}&


# wait

done

wait





