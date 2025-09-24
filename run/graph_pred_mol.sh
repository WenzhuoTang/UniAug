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
    full_settings=($@) # get cfg_name ckpt_name seed fold subset_ratio settings
    cfg_name=${full_settings[0]}
    ckpt_name=${full_settings[1]}
    seed=${full_settings[2]}
    fold=${full_settings[3]}
    subset_ratio=${full_settings[4]}
    settings=${full_settings[@]:5} # extra settings

    extra_config="augment.subset_ratio=${subset_ratio} "
    extra_args="--fold ${fold} "

    ### modifying
    # extra_config+="augment.num_repeats=1 "
    # extra_config+="augment.num_repeats=2 "
    # extra_config+="augment.num_repeats=5 "
    # extra_config+="augment.num_repeats=10 "
    extra_config+="augment.num_repeats=20 "
    # extra_config+="augment.num_repeats=32 "

    extra_config+="augment.replace_flag=True "
    # extra_config+="augment.replace_flag=False "

    # extra_config+="model.edge_feat=True data.add_self_loop=False "
    extra_config+="model.edge_feat=False "

    # extra_args+="--prefix aug_ep_5000 "
    # extra_config+="augment.guidance_config.num_epochs=5000 "
    
    ## training
    # extra_args+="--prefix epochs-50 "
    # extra_config+="train.epochs=50 "

    # extra_config+="train.selection=best "
    # extra_config+="train.selection=last "

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

    extra_args+="--prefix aug_val_aug_test "
    extra_config+="augment.augment_valid=True augment.augment_test=True "

    ## guidance
    # extra_args+="--prefix step_size-5.0 "
    # extra_config+="augment.sample.step_size=5.0 "

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
    # extra_config+="augment.sample.stability_coef=10 augment.sample.num_steps=10 "

    # extra_args+="--prefix coef-100_step-5 "
    # extra_config+="augment.sample.stability_coef=100 augment.sample.step_size=5.0 "

    # extra_args+="--prefix coef-100_sam_steps-64 "
    # extra_config+="augment.sample.num_sample_steps=64 augment.sample.stability_coef=100 "

    # extra_args+="--prefix aug_val_aug_test_sam_steps-64 "
    # extra_config+="augment.sample.num_sample_steps=64 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix aug_val_aug_test_coef-10 "
    # extra_config+="augment.sample.stability_coef=10 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix aug_val_aug_test_step-5.0 "
    # extra_config+="augment.sample.step_size=5.0 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix aug_val_aug_test_n_step-10 "
    # extra_config+="augment.sample.num_steps=10 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix val_test_coef-100_step-5.0 "
    # extra_config+="augment.sample.step_size=5.0 augment.sample.stability_coef=100 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix val_test_aug_ep_100 "
    # extra_config+="augment.guidance_config.num_epochs=100 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix val_test_sam_ste-64_coef-100 "
    # extra_config+="augment.sample.num_sample_steps=64 augment.sample.stability_coef=100 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix val_test_sam_ste-64_step-5 "
    # extra_config+="augment.sample.num_sample_steps=64 augment.sample.step_size=5.0 "
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
        extra_config+=""
    fi

    config_file="config/graph_pred_mol/${cfg_name}.yaml"

    # script="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    script="OPENBLAS_NUM_THREADS=1 "
    script+="python3 graph_pred_mol.py --config ${config_file} --seed ${seed} ${extra_args} ${settings[@]} ${extra_config} "

    echo ${script} && [[ $TEST_FLAG == 0 ]] && eval ${script}
}

# AUG_FLAG=0 CUDA_VISIBLE_DEVICES=0 bash run/graph_pred_mol.sh
# AUG_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/graph_pred_mol.sh
# AUG_FLAG=1 TRAIN_GUI_FLAG=0 CUDA_VISIBLE_DEVICES=0 bash run/graph_pred_mol.sh
# AUG_FLAG=1 TRAIN_GUI_FLAG=1 NEG_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/graph_pred_mol.sh
# AUG_FLAG=1 TRAIN_GUI_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/graph_pred_mol.sh

fold=0

echo fold=${fold}

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

seeds=(0)

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

for seed in ${seeds[@]}; do

echo seed=${seed}

#### full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide

### classification
## small ones
# launch ogbg-molbace full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch ogbg-molbbbp full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch ogbg-molsider full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&

# launch ogbg-molesol full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch ogbg-molfreesolv full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&

## medium ones
# launch ogbg-moltox21 full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&

launch ogbg-mollipo full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&

## large ones
# launch ogbg-molhiv full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&

## archived (too much performance gap or cannot load)
# launch ogbg-molclintox full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch ogbg-moltoxcast full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch ogbg-molmuv full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&


# wait

done

wait