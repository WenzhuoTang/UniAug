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
NEG_FLAG=${NEG_FLAG:-0}
# --------------------

launch () {
    full_settings=($@)
    cfg_name=${full_settings[0]}
    ckpt_name=${full_settings[1]}
    seed=${full_settings[2]}
    fold=${full_settings[3]}
    subset_ratio=${full_settings[4]}
    settings=${full_settings[@]:5}

    extra_config="augment.subset_ratio=${subset_ratio} "
    extra_args="--fold ${fold} "

    ### modifying
    # extra_config+="augment.num_repeats=1 "
    # extra_config+="augment.num_repeats=2 "
    # extra_config+="augment.num_repeats=5 "
    # extra_config+="augment.num_repeats=10 "
    # extra_config+="augment.num_repeats=20 "

    # extra_config+="augment.replace_flag=True "
    # extra_config+="augment.replace_flag=False "

    ## num_sample_steps
    # extra_args+="--prefix n_sample_steps-64 "
    # extra_config+="augment.sample.num_sample_steps=64 "

    # extra_args+="--prefix n_sample_steps-32 "
    # extra_config+="augment.sample.num_sample_steps=32 "
    
    # extra_args+="--prefix n_sample_steps-16 "
    # extra_config+="augment.sample.num_sample_steps=16 "

    # extra_args+="--prefix n_sample_steps-8 "
    # extra_config+="augment.sample.num_sample_steps=8 "

    # extra_args+="--prefix n_sample_steps-4 "
    # extra_config+="augment.sample.num_sample_steps=4 "

    ## augment
    # extra_args+="--prefix aug_val "
    # extra_config+="augment.augment_valid=True "

    # extra_args+="--prefix aug_test "
    # extra_config+="augment.augment_test=True "

    # extra_args+="--prefix aug_val_aug_test "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    ## guidance
    # extra_args+="--prefix step_size-5.0 "
    # extra_config+="augment.sample.step_size=5.0 "

    # extra_args+="--prefix n_step-10 "
    # extra_config+="augment.sample.num_steps=10 "

    # extra_args+="--prefix coef-10 "
    # extra_config+="augment.sample.stability_coef=10 "

    ## guidance head
    # extra_args+="--prefix head_l1 "
    # extra_config+="augment.guidance_config.guidance.num_layers=1 "

    # extra_args+="--prefix head_dp0.5 "
    # extra_config+="augment.guidance_config.guidance.dropout=0.5 "

    # extra_config+="augment.guidance_config.diffusion.guidance_type=node_degree "
    # extra_config+="augment.guidance_config.guidance.target=node "

    ## combined
    # extra_args+="--prefix coef-10_n_step-10 "
    # extra_config+="augment.sample.stability_coef=10 augment.sample.num_steps=10 "

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

    if [[ $NEG_FLAG == 1 ]]; then
        extra_args+="--neg_guide t "
    fi

    if [[ $AUG_FLAG == 1 ]]; then
        extra_args+="--augment t "
        ckpt_path="checkpoints/network_repository/"

        extra_config+="data.extract_attributes=True "

        if [[ $ckpt_name == nr_gnn_gt_l4_deg_lin ]]; then
            ckpt_path+="nr_gnn_gt_l4_deg_lin-r.1-Feb21-21:45:49.pth"

        elif [[ $ckpt_name == nr_snap_gnn_gt_l4_deg_lin ]]; then
            ckpt_path="checkpoints/network_repository_large_network_repository_snap/"
            ckpt_path+="nr_snap_gnn_gt_l4_deg_lin-r.1-Feb22-19:42:52_6000.pth"

        elif [[ $ckpt_name == full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ]]; then
            ckpt_path+="full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide-r.1-Mar08-02:01:14.pth"
        elif [[ $ckpt_name == full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ]]; then
            ckpt_path+="full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide-r.1-Mar29-21:15:57.pth"
        
        elif [[ $ckpt_name == full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_1000 ]]; then
            ckpt_path+="full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide-r.1-Mar26-21:29:46_1000.pth"
        
        elif [[ $ckpt_name == full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_3000 ]]; then
            ckpt_path+="full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide-r.1-Mar26-21:29:46_3000.pth"
        
        elif [[ $ckpt_name == full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_5000 ]]; then
            ckpt_path+="full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide-r.1-Mar26-21:29:46_5000.pth"
        
        elif [[ $ckpt_name == full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_7000 ]]; then
            ckpt_path+="full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide-r.1-Mar26-21:29:46_7000.pth"

        else
            echo Unknown ckpt name $ckpt_name && exit 1
        fi
        extra_config+="augment.ckpt_path=${ckpt_path} "

    else
        extra_config=""
    fi

    config_file="config/graph_pred/${cfg_name}.yaml"

    script="python graph_pred.py --config ${config_file} --seed ${seed} ${extra_args} ${settings[@]} ${extra_config} "

    echo ${script} && [[ $TEST_FLAG == 0 ]] && eval ${script}
}

# AUG_FLAG=0 CUDA_VISIBLE_DEVICES=0 bash run/graph_pred.sh
# AUG_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/graph_pred.sh
# AUG_FLAG=1 TRAIN_GUI_FLAG=0 CUDA_VISIBLE_DEVICES=0 bash run/graph_pred.sh
# AUG_FLAG=1 TRAIN_GUI_FLAG=1 NEG_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/graph_pred.sh
# AUG_FLAG=1 TRAIN_GUI_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/graph_pred.sh

seed=0

echo seed=${seed}

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

# folds=(0)

folds=(0 1 2 3 4 5 6 7 8 9)

# folds=(0 1 2 3 4)
# folds=(5 6 7 8 9)

# folds=(0 1 2 3)
# folds=(4 5 6 7)
# folds=(8 9)

# folds=(0 1 2)
# folds=(3 4 5)
# folds=(6 7)
# folds=(8 9)

for fold in ${folds[@]}; do

echo fold=${fold}

### nr_gnn_gt_l4_deg_emb
# launch ogbg-ppa nr_gnn_gt_l4_deg_emb ${seed} ${fold} ${thres}&

### nr_gnn_gt_l4_deg_lin
# launch Enzymes nr_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&
# launch NCI1 nr_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&
# launch Proteins nr_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&

# launch IMDB_binary nr_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&
# launch IMDB_multi nr_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&

# launch Collab nr_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&
# launch DD nr_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&
# launch Reddit_binary nr_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&
# launch Reddit_multi_5k nr_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&
# launch Reddit_multi_12k nr_gnn_gt_l4_deg_lin-0_deg_lin_self_guide ${seed} ${fold} ${thres}&

### nr_snap_gnn_gt_l4_deg_emb
## no guidance
# launch DD nr_snap_gnn_gt_l4_deg_emb ${seed} ${fold} ${thres}&

### nr_snap_gnn_gt_l4_deg_lin
## no guidance
# launch DD nr_snap_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&



### full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide
# launch Enzymes full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch NCI1 full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch Proteins full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&

# launch IMDB_binary full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch IMDB_multi full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&

# launch Collab full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch DD full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch Reddit_binary full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch Reddit_multi_5k full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch Reddit_multi_12k full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&



### full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide
# launch Enzymes full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch NCI1 full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch Proteins full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&

# launch IMDB_binary full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
launch IMDB_multi full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&

# launch Collab full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch DD full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch Reddit_binary full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch Reddit_multi_5k full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch Reddit_multi_12k full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&


### nr_snap_gnn_gt_l4_deg_lin
# launch Enzymes nr_snap_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&
# launch Proteins nr_snap_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&
# launch IMDB_binary nr_snap_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&
# launch IMDB_multi nr_snap_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&


### full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_1000
# launch Enzymes full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_1000 ${seed} ${fold} ${thres}&
# launch Proteins full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_1000 ${seed} ${fold} ${thres}&
# launch IMDB_binary full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_1000 ${seed} ${fold} ${thres}&
# launch IMDB_multi full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_1000 ${seed} ${fold} ${thres}&

### full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_3000
# launch Enzymes full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_3000 ${seed} ${fold} ${thres}&
# launch Proteins full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_3000 ${seed} ${fold} ${thres}&
# launch IMDB_binary full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_3000 ${seed} ${fold} ${thres}&
# launch IMDB_multi full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_3000 ${seed} ${fold} ${thres}&

### full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_5000
# launch Enzymes full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_5000 ${seed} ${fold} ${thres}&
# launch Proteins full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_5000 ${seed} ${fold} ${thres}&
# launch IMDB_binary full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_5000 ${seed} ${fold} ${thres}&
# launch IMDB_multi full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_5000 ${seed} ${fold} ${thres}&

### full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_7000
# launch Enzymes full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_7000 ${seed} ${fold} ${thres}&
# launch Proteins full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_7000 ${seed} ${fold} ${thres}&
# launch IMDB_binary full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_7000 ${seed} ${fold} ${thres}&
# launch IMDB_multi full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide_7000 ${seed} ${fold} ${thres}&

# wait

done

wait




### iterate over seeds

# seeds=(0)

# seeds=(0 1 2 3 4 5 6 7 8 9)

# seeds=(0 1 2 3 4)
# seeds=(5 6 7 8 9)

# seeds=(0 1 2 3)
# seeds=(4 5 6 7)
# seeds=(8 9)

# seeds=(0 1 2)
# seeds=(3 4 5)
# seeds=(6 7)
# seeds=(8 9)


# for seed in ${seeds[@]}; do

# echo seed=${seed}

### nr_gnn_gt_l4_deg_lin
# launch Reddit_multi_12k nr_gnn_gt_l4_deg_lin ${seed} ${fold} ${thres}&

### full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide
# launch Reddit_multi_12k full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${thres}&

# wait

# done

# wait





