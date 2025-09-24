#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########

#SBATCH -e ./log/nc_hete.err
#SBATCH -o ./log/nc_hete.out
#SBATCH --time=4:00:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH -c 1           # number of CPUs (or cores) per task (same as -c
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name nc_hete # you can give your job a name for easier identification (same as -J)

source ~/.bashrc
source ~/anaconda3/bin/activate graph

trap "echo ERROR && exit 1" ERR

# --------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TEST_FLAG=${TEST_FLAG:-0}
AUG_FLAG=${AUG_FLAG:-1}
TRAIN_GUI_FLAG=${TRAIN_GUI_FLAG:-1}
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
    extra_config+="augment.num_repeats=4 "
    # extra_config+="augment.num_repeats=2 "
    # extra_config+="augment.num_repeats=5 "
    # extra_config+="augment.num_repeats=10 "
    # extra_config+="augment.num_repeats=20 "

    # extra_config+="augment.replace_flag=True "
    extra_config+="augment.replace_flag=False "

    extra_config+="data.full_to_undirected=True "
    # extra_config+="data.full_to_undirected=False "

    extra_config+="data.sub_to_undirected=True "
    # extra_config+="data.sub_to_undirected=False "

    # extra_config+="augment.nodes_max=8e3 "
    # extra_config+="augment.edges_max=1e5 "

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
    # extra_args+="--prefix coef-10_n_step-10 "
    # extra_config+="augment.sample.stability_coef=10 augment.sample.num_steps=10 "

    # extra_args+="--prefix coef-10_n_sam_steps-16 "
    # extra_config+="augment.sample.num_sample_steps=16 augment.sample.stability_coef=10 "

    # extra_args+="--prefix aug_val_aug_test_coef-100 "
    # extra_config+="augment.sample.stability_coef=100 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix aug_val_aug_test_step_size-5.0 "
    # extra_config+="augment.sample.step_size=5.0 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix aug_val_aug_test_n_step-10 "
    # extra_config+="augment.sample.num_steps=10 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix aug_val_aug_test_coef-10_n_sam_steps-16 "
    # extra_config+="augment.sample.num_sample_steps=16 augment.sample.stability_coef=10 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    if [[ $TRAIN_GUI_FLAG == 1 ]]; then
        extra_args+="--train_guidance t "
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

    script="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python node_class_subgraph.py"
    script+=" --config ${config_file} --seed ${seed} ${extra_args} ${settings[@]} ${extra_config} "

    echo ${script} && [[ $TEST_FLAG == 0 ]] && eval ${script}
}

# AUG_FLAG=0 CUDA_VISIBLE_DEVICES=0 bash run/node_class_subgraph_fold.sh
# AUG_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/node_class_subgraph_fold.sh
# AUG_FLAG=1 TRAIN_GUI_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/node_class_subgraph_fold.sh

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

# folds=(0 1 2 3 4 5 6 7 8 9)

# folds=(0 1 2 3 4)
# folds=(5 6 7 8 9)

# folds=(0 1 2 3)
# folds=(4 5 6 7)
# folds=(8 9)

folds=(0)
# folds=(1)
# folds=(2)
# folds=(3)
# folds=(4)
# folds=(5)
# folds=(6)
# folds=(7)
# folds=(8)
# folds=(9)

for fold in ${folds[@]}; do

echo fold=${fold}

#### full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide
# launch cornell full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch texas full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch wisconsin full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&

# launch chameleon full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch squirrel full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&

# launch actor full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch deezer_europe full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&





#### full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide
# launch cornell full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch texas full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch wisconsin full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&
# launch actor full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&

## 1 per time
# launch actor full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}& # 73g
# launch chameleon full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}& # 37g
# launch squirrel full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}& # 

## 5 folds
# launch deezer_europe full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide ${seed} ${fold} ${subset_ratio}&

# wait

done

wait

