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
DEG_EMB_FLAG=${DEG_EMB_FLAG:-0}
DEG_LIN_FLAG=${DEG_LIN_FLAG:-0}
GUIDE_FLAG=${GUIDE_FLAG:-0}
DDP_FLAG=${DDP_FLAG:-0}
# --------------------

launch () {
    full_settings=($@)
    cfg_name=${full_settings[0]}
    prefix=${full_settings[1]}
    seed=${full_settings[2]}
    settings=${full_settings[@]:3}

    extra_config=""

    ## modifying
    # prefix+="_dnnm-50" 
    # extra_config+="data.dense_num_nodes_max=50 "  # 0 | 50 | 200

    # prefix+="_load6000"
    # extra_config+="diffusion.ckpt_path=checkpoints/network_repository/full_nr_github_gnn_gt_l4_dnnm-50_deg_lin_self_guide-r.1-Mar26-21:29:46_6000.pth "
    # extra_config+="train.num_epochs=4000 "

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

# num layers -------------------------------------------------------------------

# layers_list=(1 2 3 4 5)

# layers_list=(1 2 3 4)
# layers_list=(5)

# layers_list=(1 2 3)
# layers_list=(4 5)

# layers_list=(1 2)
# layers_list=(3 4)
# layers_list=(5)

# layers_list=(6)
layers_list=(4)
# layers_list=(3)
for layers in ${layers_list[@]}; do

# layers=1
echo num_layers=${layers}

## gnn_gcn
# launch comm_gnn comm_gnn_gcn_l${layers} ${seed} model.gnn_type=gcn model.num_layers=${layers}&
# launch ego_gnn ego_gnn_gcn_l${layers} ${seed} model.gnn_type=gcn model.num_layers=${layers}&
# launch full_nr_gnn full_nr_gnn_gcn_l${layers} ${seed} model.gnn_type=gcn model.num_layers=${layers}&

## gnn_sage
# launch comm_gnn comm_gnn_sage_l${layers} ${seed} model.gnn_type=sage model.num_layers=${layers}&
# launch ego_gnn ego_gnn_sage_l${layers} ${seed} model.gnn_type=sage model.num_layers=${layers}&

## gnn_gat
# launch comm_gnn comm_gnn_gat_l${layers} ${seed} model.gnn_type=gat model.num_layers=${layers}&
# launch ego_gnn ego_gnn_gat_l${layers} ${seed} model.gnn_type=gat model.num_layers=${layers}&

## gnn_gt
# launch comm_gnn comm_gnn_gt_l${layers} ${seed} model.gnn_type=gt model.num_layers=${layers}&
# launch ego_gnn ego_gnn_gt_l${layers} ${seed} model.gnn_type=gt model.num_layers=${layers}&
# launch nr_gnn nr_gnn_gt_l${layers} ${seed} model.gnn_type=gt model.num_layers=${layers}&
# launch nr_snap_gnn nr_snap_gnn_gt_l${layers} ${seed} model.gnn_type=gt model.num_layers=${layers}&
# launch full_nr_gnn full_nr_gnn_gt_l${layers} ${seed} model.gnn_type=gt model.num_layers=${layers}&
# launch full_nr_github_gnn full_nr_github_gnn_gt_l${layers} ${seed} model.gnn_type=gt model.num_layers=${layers}&
# launch full_nr_reddit2k_gnn full_nr_reddit2k_gnn_gt_l${layers} ${seed} model.gnn_type=gt model.num_layers=${layers}&

## 8 clusters
# cluster=0  # 0 - 7
# echo cluster=${cluster}
# launch nr_snap_clus_gnn nr_snap_clus_gnn_gt_clus${cluster}_l${layers} ${seed} model.gnn_type=gt model.num_layers=${layers} data.cluster=${cluster}&
## batch_size: 0-8, 1-4, 2-1, 3-6, 4-1 (37g), 5-1(37g), 6-6, 7-1

## 8 clusters
# cluster=6  # 0, 1, 3, 4, 6
# echo cluster=${cluster}
# launch nr_clus_gnn nr_clus_gnn_gt_clus${cluster}_l${layers} ${seed} model.gnn_type=gt model.num_layers=${layers} data.cluster=${cluster}&
## batch_size: 0-16 (43g), 1-6 (22g), 3-16 (40g), 4-1 (16g), 6-16 (41g)

## tgnn
# launch comm_tgnn comm_tgnn_l${layers} ${seed} model.num_layers=${layers}&
# launch ego_tgnn ego_tgnn_l${layers} ${seed} model.num_layers=${layers}&
# launch nr_tgnn nr_tgnn_l${layers} ${seed} model.num_layers=${layers}&
# launch nr_snap_tgnn nr_snap_tgnn_l${layers} ${seed} model.num_layers=${layers}&

## sta
# launch comm_sta comm_sta ${seed}&
# launch ego_sta ego_sta ${seed&

## zinc
launch zinc zinc_gt_l${layers} ${seed} model.gnn_type=gt model.num_layers=${layers}&

done

wait

# done




