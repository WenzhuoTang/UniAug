#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########

#SBATCH -e ./log/graph_pred_reddit5k.err
#SBATCH -o ./log/graph_pred_reddit5k.out
#SBATCH --time=4:00:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH -c 1           # number of CPUs (or cores) per task (same as -c
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name gp_red5k # you can give your job a name for easier identification (same as -J)

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
    thres=${full_settings[4]}
    settings=${full_settings[@]:5}

    extra_config=""
    extra_args="--fold ${fold} --thres ${thres} "

    ### modifying
    extra_config+="augment.num_repeats=1 "
    # extra_config+="augment.num_repeats=2 "
    # extra_config+="augment.num_repeats=5 "
    # extra_config+="augment.num_repeats=10 "
    # extra_config+="augment.num_repeats=20 "

    extra_config+="augment.replace_flag=True "
    # extra_config+="augment.replace_flag=False "

    ## num_sample_steps
    # extra_args+="--prefix n_sample_steps-64 "
    # extra_config+="augment.sample.num_sample_steps=64 "

    # extra_args+="--prefix n_sample_steps-32 "
    # extra_config+="augment.sample.num_sample_steps=32 "
    
    extra_args+="--prefix n_sample_steps-16 "
    extra_config+="augment.sample.num_sample_steps=16 "

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
    # extra_args+="--prefix step_size-0.5 "
    # extra_config+="augment.sample.step_size=0.5 "

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

    # extra_args+="--prefix aug_val_aug_test_n_sam_steps-16 "
    # extra_config+="augment.sample.num_sample_steps=16"
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix aug_val_aug_test_step_size-1.0_n_sam_steps-16 "
    # extra_config+="augment.sample.num_sample_steps=16 augment.sample.step_size=1.0 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

    # extra_args+="--prefix aug_val_aug_test_coef-10_n_sam_steps-16 "
    # extra_config+="augment.sample.num_sample_steps=16 augment.sample.stability_coef=10 "
    # extra_config+="augment.augment_valid=True augment.augment_test=True "

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

        extra_config+="data.extract_attributes=True "

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
            ckpt_path+="nr_snap_gnn_gt_l4_deg_lin-r.1-Feb22-19:42:52_6000.pth"

        elif [[ $ckpt_name == full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ]]; then
            ckpt_path+="full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide-r.1-Mar08-02:01:14.pth"
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
# AUG_FLAG=1 TRAIN_GUI_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/graph_pred.sh

thres=None

# thres=0.5
# thres=0.6
# thres=0.7
# thres=0.8

# thres=0.9
# thres=0.99
# thres=0.999

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


seed=0

echo seed=${seed}

# folds=(0)

# folds=(0 1 2 3 4 5 6 7 8 9)

# folds=(0 1 2 3 4)
# folds=(5 6 7 8 9)

# folds=(0 1 2 3)
# folds=(4 5 6 7)
# folds=(8 9)

# folds=(0 1 2)
# folds=(3 4 5)
# folds=(6 7)
# folds=(8 9)

# folds=(0 1)
# folds=(2 3)
# folds=(4 5)
# folds=(6 7)
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

### full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide
launch Reddit_multi_5k full_nr_gnn_gt_l4_dnnm-0_deg_lin_self_guide ${seed} ${fold} ${thres}&


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





