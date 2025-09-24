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
TRAIN_GUI_FLAG=${TRAIN_GUI_FLAG:-0}
PREDICT_FLAG=${PREDICT_FLAG:-0}
PREDICT_ALL=${PREDICT_ALL:-0}
# --------------------

launch () {
    full_settings=($@)
    cfg_name=${full_settings[0]}
    ckpt_name=${full_settings[1]}
    seed=${full_settings[2]}
    subset_ratio=${full_settings[3]}
    settings=${full_settings[@]:4}

    extra_config="augment.subset_ratio=${subset_ratio} "
    extra_args=""

    ### modifying
    extra_config+="augment.num_repeats=5 "
    # extra_config+="augment.num_repeats=2 "
    # extra_config+="augment.num_repeats=5 "
    # extra_config+="augment.num_repeats=10 "
    # extra_config+="augment.num_repeats=20 "

    extra_config+="augment.replace_flag=True "
    # extra_config+="augment.replace_flag=False "

    ## training
    extra_config+="train.selection=best "
    # extra_config+="train.selection=last "

    # extra_args+="--prefix epochs-50 "
    # extra_config+="train.epochs=50 train.patience=100"

    ## num_sample_steps
    # extra_args+="--prefix n_sample_steps-64 "
    # extra_config+="augment.sample.num_sample_steps=64 "

    ## augment
    # extra_args+="--prefix aug_val "
    # extra_config+="augment.augment_valid=True "

    # extra_args+="--prefix aug_test "
    # extra_config+="augment.augment_test=True "

    # extra_args+="--prefix aug_val_aug_test "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    ## guidance
    # extra_args+="--prefix step_size-10.0 "
    # extra_config+="augment.sample.step_size=10.0 "

    # extra_args+="--prefix n_step-10 "
    # extra_config+="augment.sample.num_steps=10 "

    # extra_args+="--prefix coef-100 "
    # extra_config+="augment.sample.stability_coef=100 "

    ## guidance head
    # extra_args+="--prefix head_l1 "
    # extra_config+="augment.guidance_config.guidance.num_layers=1 "

    # extra_args+="--prefix head_dp0.5 "
    # extra_config+="augment.guidance_config.guidance.dropout=0.5 "

    # extra_config+="augment.guidance_config.diffusion.guidance_type=node_degree "
    # extra_config+="augment.guidance_config.guidance.target=node "

    ## combined
    # extra_args+="--prefix coef-100_n_step-10 "
    # extra_config+="augment.sample.stability_coef=100 augment.sample.num_steps=10 "

    # extra_args+="--prefix coef-10_n_sam_steps-16 "
    # extra_config+="augment.sample.num_sample_steps=16 augment.sample.stability_coef=10 "

    # extra_args+="--prefix aug_val_aug_test_coef-10 "
    # extra_config+="augment.sample.stability_coef=10 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix aug_val_aug_test_step_size-0.5 "
    # extra_config+="augment.sample.step_size=0.5 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix aug_val_aug_test_n_step-20 "
    # extra_config+="augment.sample.num_steps=20 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix aug_val_aug_test_coef-10_n_sam_steps-16 "
    # extra_config+="augment.sample.num_sample_steps=16 augment.sample.stability_coef=10 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    if [[ $TRAIN_GUI_FLAG == 1 ]]; then
        extra_args+="--train_guidance t "
    fi

    if [[ $PREDICT_FLAG == 1 ]]; then
        extra_args+="--save_prediction t "
    fi

    if [[ $PREDICT_ALL == 1 ]]; then
        extra_args+="--predict_all t "
    fi

    if [[ $AUG_FLAG == 1 ]]; then
        extra_args+="--augment t "
        ckpt_path="checkpoints/network_repository/"

        extra_config+="data.extract_attributes=True "

        if [[ $ckpt_name == full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ]]; then
            ckpt_path+="full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide-r.1-Mar08-02:01:14.pth"
        elif [[ $ckpt_name == full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ]]; then
            ckpt_path+="full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide-r.1-Mar29-21:15:57.pth"
        else
            echo Unknown ckpt name $ckpt_name && exit 1
        fi
        extra_config+="augment.ckpt_path=${ckpt_path} "

    else
        extra_config=""
    fi

    config_file="config/node_class_subgraph/${cfg_name}.yaml"

    script="python node_class_subgraph.py --config ${config_file} --seed ${seed} ${extra_args} ${extra_config} ${settings[@]} "

    echo ${script} && [[ $TEST_FLAG == 0 ]] && eval ${script}
}

# AUG_FLAG=0 CUDA_VISIBLE_DEVICES=0 bash run/node_class_subgraph.sh
# AUG_FLAG=0 PREDICT_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/node_class_subgraph.sh
# AUG_FLAG=0 PREDICT_FLAG=1 PREDICT_ALL=1 CUDA_VISIBLE_DEVICES=0 bash run/node_class_subgraph.sh

# AUG_FLAG=1 TRAIN_GUI_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/node_class_subgraph.sh
# AUG_FLAG=1 TRAIN_GUI_FLAG=1 PREDICT_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/node_class_subgraph.sh
# AUG_FLAG=1 TRAIN_GUI_FLAG=1 PREDICT_FLAG=1 PREDICT_ALL=1 CUDA_VISIBLE_DEVICES=0 bash run/node_class_subgraph.sh


subset_ratio=1
# subset_ratio=0.9
# subset_ratio=0.8
# subset_ratio=0.7
# subset_ratio=0.6
# subset_ratio=0.5
# subset_ratio=0.4
# subset_ratio=0.3
# subset_ratio=0.2
# subset_ratio=0.1

echo subset_ratio=${subset_ratio}

# seeds=(1)

seeds=(0 1 2 3 4 5 6 7 8 9)

# seeds=(0 1 2 3 4)
# seeds=(5 6 7 8 9)

# seeds=(0 1 2 3)
# seeds=(4 5 6 7)
# seeds=(8 9)

# seeds=(0 1 2 3 4 5)
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

### full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide
# launch cora full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${subset_ratio}&
# launch citeseer full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${subset_ratio}&
# launch pubmed full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${subset_ratio}&



### full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide
# launch cora full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${subset_ratio}&
launch citeseer full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${subset_ratio}&
# launch pubmed full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${subset_ratio}&


# wait

done

wait