#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########

#SBATCH -e ./log/full_nr_reddit2k_gnn.err
#SBATCH -o ./log/full_nr_reddit2k_gnn.out
#SBATCH --time=100:00:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH -c 1           # number of CPUs (or cores) per task (same as -c
#SBATCH --gres=gpu:a100:3
#SBATCH --mem=200G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name reddit2k # you can give your job a name for easier identification (same as -J)

source ~/.bashrc
source ~/anaconda3/bin/activate graph

trap "echo ERROR && exit 1" ERR

# --------------------
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TEST_FLAG=${TEST_FLAG:-0}
DEG_EMB_FLAG=${DEG_EMB_FLAG:-0}
DEG_LIN_FLAG=${DEG_LIN_FLAG:-1}
GUIDE_FLAG=${GUIDE_FLAG:-1}
DDP_FLAG=${DDP_FLAG:-1}
# --------------------

launch () {
    full_settings=($@)
    cfg_name=${full_settings[0]}
    prefix=${full_settings[1]}
    seed=${full_settings[2]}
    settings=${full_settings[@]:3}

    extra_config=""

    ## modifying
    prefix+="_dnnm-50" 
    extra_config+="data.dense_num_nodes_max=50 "  # 0 | 50 | 200

    if [[ $DEG_EMB_FLAG == 1 ]]; then
        prefix+="_deg_emb"
        extra_config+="model.emb_type=degree_embedding data.feature_flag=False"
    elif [[ $DEG_LIN_FLAG == 1 ]]; then
        prefix+="_deg_lin"
        extra_config+="model.emb_type=degree_linear data.feature_flag=False"
    else
        extra_config+="model.emb_type=feat_linear data.feature_flag=True"
    fi

    if [[ $GUIDE_FLAG == 1 ]]; then
        prefix+="_self_guide"
        extra_config+=" model.target=GuidedGNN"
    fi

    if [[ $DDP_FLAG == 1 ]]; then
        fname="pretrain_ddp.py"
    else
        fname="pretrain.py"
    fi

    config_file="pretrain_config/binary/${cfg_name}.yaml"

    script="python ${fname} --config ${config_file}"
    script+=" --prefix ${prefix} ${settings[@]} ${extra_config} "

    echo ${script} && [[ $TEST_FLAG == 0 ]] && eval ${script}
}


# DEG_EMB_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/pretrain.sh
# DEG_LIN_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/pretrain.sh
# DEG_LIN_FLAG=1 GUIDE_FLAG=1 CUDA_VISIBLE_DEVICES=0 bash run/pretrain.sh
# DEG_LIN_FLAG=1 GUIDE_FLAG=1 DDP_FLAG=1 CUDA_VISIBLE_DEVICES=0,1 bash run/pretrain.sh

seed=1
echo seed=${seed}


layers_list=(4)
for layers in ${layers_list[@]}; do

echo num_layers=${layers}

## gnn_gt
# launch full_nr_gnn full_nr_gnn_gt_l${layers} ${seed} model.gnn_type=gt model.num_layers=${layers}&
# launch full_nr_github_gnn full_nr_github_gnn_gt_l${layers} ${seed} model.gnn_type=gt model.num_layers=${layers}&
launch full_nr_reddit2k_gnn full_nr_reddit2k_gnn_gt_l${layers} ${seed} model.gnn_type=gt model.num_layers=${layers}&

done

wait