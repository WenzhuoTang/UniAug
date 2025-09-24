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
NAME=nc_tune
AUG_FLAG=${AUG_FLAG:-1}
ORIG_FEAT_FLAG=${ORIG_FEAT_FLAG:-0}
TEST_FLAG=${TEST_FLAG:-0}
TASK=${TASK:-"NC"}
# --------------------

HOMEDIR=$(dirname $(dirname $(realpath $0)))
cd $HOMEDIR
echo HOMEDIR=$HOMEDIR

launch () {
    full_settings=($@)
    name=${full_settings[0]}
    subgraph_type=${full_settings[1]}
    pre_data=${full_settings[2]}
    tag=${full_settings[3]}
    seed=${full_settings[4]}
    settings=${full_settings[@]:5}

    if [[ $TASK == NC ]]; then
        cfg_prefix=nc
    elif [[ $TASK == LP ]]; then
        cfg_prefix=LP
    else
        echo Unknown task ${TASK} && exit 1
    fi

    if [[ $name == cora ]]; then
        dataset_name=cora
        config_file="config/cora_${cfg_prefix}-${subgraph_type}.yaml"
        feat_dim=1433
    elif [[ $name == citeseer ]]; then
        dataset_name=citeseer
        config_file="config/citeseer_${cfg_prefix}-${subgraph_type}.yaml"
        feat_dim=3703
    elif [[ $name == pubmed ]]; then
        dataset_name=pubmed
        config_file="config/pubmed_${cfg_prefix}-${subgraph_type}.yaml"
        feat_dim=500
    else
        echo Unknown dataset ${name} && exit 1
    fi

    if [[ $ORIG_FEAT_FLAG == 1 ]]; then
        extra_config="data.feature_types=null data.stru_feat_principle=null model.input_dim=${feat_dim}"
        tag="orig_feat_${tag}"
    else
        extra_config=""
    fi

    if [[ $pre_data == network_repository ]]; then
        ckpt_path="checkpoints/network_repository/3_hop-max_200-Nov30-17:40:05_3200.pth"
    elif [[ $pre_data == cora ]]; then
        ckpt_path="checkpoints/cora/3_hop-max_200-Nov29-21:47:54.pth"
    elif [[ $pre_data == planetoid ]]; then
        ckpt_path="checkpoints/planetoid/3_hop-max_200-Nov30-17:38:46_5000.pth"
    else
        echo Unknown pretrain dataset ${pre_data} && exit 1
    fi

    if [[ $AUG_FLAG == 1 ]]; then
        aug=True
    else
        aug=False
    fi

    script="python main.py --config ${config_file} --seed ${seed} --augment ${aug}"
    script+=" --prefix ${name}_${tag}_r.${seed} augment.ckpt_path=${ckpt_path} ${extra_config} ${settings[@]}"

    echo ${script} && [[ $TEST_FLAG == 0 ]] && eval ${script}
}


# 5 repeats -------------------------------------------------------------------
# seeds=(0 1 2 3 4)

# for seed in ${seeds[@]}; do

# echo seed=${seed}

# launch cora network_repository default ${seed} &
# launch citeseer network_repository default ${seed} &
# launch pubmed network_repository default ${seed} &

# done


# -----------------------------------------------------------------------------

# Tuning -------------------------------------------------------------------
seed=1

# model & training -------------------------------------------------------------------

### test
# launch cora ego network_repository test ${seed} train.epochs=10 &
# launch citeseer ego network_repository test ${seed} train.epochs=10 &
# launch pubmed ego network_repository test ${seed} train.epochs=10 &

# launch cora rw network_repository test ${seed} train.epochs=10 &
# launch citeseer rw network_repository test ${seed} train.epochs=10 &
# launch pubmed rw network_repository test ${seed} train.epochs=10 &

### default
# launch cora ego network_repository default ${seed} &
# launch citeseer ego network_repository default ${seed} &
# launch pubmed ego network_repository default ${seed} &

# launch cora rw network_repository default ${seed} &
# launch citeseer rw network_repository default ${seed} &
# launch pubmed rw network_repository default ${seed} &

### full_subgraph
# launch cora ego network_repository full_subgraph ${seed} data.max_node_num=null &
# launch citeseer ego network_repository full_subgraph ${seed} data.max_node_num=null &
# launch pubmed ego network_repository full_subgraph ${seed} data.max_node_num=null &

# launch cora rw network_repository full_subgraph ${seed} data.max_node_num=null &
# launch citeseer rw network_repository full_subgraph ${seed} data.max_node_num=null &
# launch pubmed rw network_repository full_subgraph ${seed} data.max_node_num=null &

### num_layer
# launch cora ego network_repository num_layer=1 ${seed} model.num_layer=1 &
# launch cora ego network_repository num_layer=2 ${seed} model.num_layer=2 &
# launch cora ego network_repository num_layer=3 ${seed} model.num_layer=3 &
# launch cora ego network_repository num_layer=4 ${seed} model.num_layer=4 &

# launch cora rw network_repository num_layer=1 ${seed} model.num_layer=1 &
# launch cora rw network_repository num_layer=2 ${seed} model.num_layer=2 &
# launch cora rw network_repository num_layer=3 ${seed} model.num_layer=3 &
# launch cora rw network_repository num_layer=4 ${seed} model.num_layer=4 &

# launch citeseer ego network_repository num_layer=1 ${seed} model.num_layer=1 &
# launch citeseer ego network_repository num_layer=2 ${seed} model.num_layer=2 &
# launch citeseer ego network_repository num_layer=3 ${seed} model.num_layer=3 &
# launch citeseer ego network_repository num_layer=4 ${seed} model.num_layer=4 &

# launch citeseer rw network_repository num_layer=1 ${seed} model.num_layer=1 &
# launch citeseer rw network_repository num_layer=2 ${seed} model.num_layer=2 &
# launch citeseer rw network_repository num_layer=3 ${seed} model.num_layer=3 &
# launch citeseer rw network_repository num_layer=4 ${seed} model.num_layer=4 &

# launch pubmed ego network_repository num_layer=1 ${seed} model.num_layer=1 &
# launch pubmed ego network_repository num_layer=2 ${seed} model.num_layer=2 &
# launch pubmed ego network_repository num_layer=3 ${seed} model.num_layer=3 &
# launch pubmed ego network_repository num_layer=4 ${seed} model.num_layer=4 &

# launch pubmed rw network_repository num_layer=1 ${seed} model.num_layer=1 &
# launch pubmed rw network_repository num_layer=2 ${seed} model.num_layer=2 &
# launch pubmed rw network_repository num_layer=3 ${seed} model.num_layer=3 &
# launch pubmed rw network_repository num_layer=4 ${seed} model.num_layer=4 &

### emb_dim
# launch cora ego network_repository emb_dim=32 ${seed} model.emb_dim=32 &
# launch cora ego network_repository emb_dim=64 ${seed} model.emb_dim=64 &
# launch cora ego network_repository emb_dim=128 ${seed} model.emb_dim=128 &
# launch cora ego network_repository emb_dim=256 ${seed} model.emb_dim=256 &
# launch cora ego network_repository emb_dim=512 ${seed} model.emb_dim=512 &

# launch citeseer ego network_repository emb_dim=32 ${seed} model.emb_dim=32 &
# launch citeseer ego network_repository emb_dim=64 ${seed} model.emb_dim=64 &
# launch citeseer ego network_repository emb_dim=128 ${seed} model.emb_dim=128 &
# launch citeseer ego network_repository emb_dim=256 ${seed} model.emb_dim=256 &
# launch citeseer ego network_repository emb_dim=512 ${seed} model.emb_dim=512 &

# launch pubmed ego network_repository emb_dim=32 ${seed} model.emb_dim=32 &
# launch pubmed ego network_repository emb_dim=64 ${seed} model.emb_dim=64 &
# launch pubmed ego network_repository emb_dim=128 ${seed} model.emb_dim=128 &
# launch pubmed ego network_repository emb_dim=256 ${seed} model.emb_dim=256 &
# launch pubmed ego network_repository emb_dim=512 ${seed} model.emb_dim=512 &

### lr
# launch cora ego network_repository lr=1e-3 ${seed} train.lr=1e-3 &
# launch cora ego network_repository lr=5e-3 ${seed} train.lr=5e-3 &
# launch cora ego network_repository lr=1e-2 ${seed} train.lr=1e-2 &
# launch cora ego network_repository lr=5e-4 ${seed} train.lr=5e-4 &
# launch cora ego network_repository lr=1e-4 ${seed} train.lr=1e-4 &

# launch citeseer ego network_repository lr=1e-3 ${seed} train.lr=1e-3 &
# launch citeseer ego network_repository lr=5e-3 ${seed} train.lr=5e-3 &
# launch citeseer ego network_repository lr=1e-2 ${seed} train.lr=1e-2 &
# launch citeseer ego network_repository lr=5e-4 ${seed} train.lr=5e-4 &
# launch citeseer ego network_repository lr=1e-4 ${seed} train.lr=1e-4 &

# launch pubmed ego network_repository lr=1e-3 ${seed} train.lr=1e-3 &
# launch pubmed ego network_repository lr=5e-3 ${seed} train.lr=5e-3 &
# launch pubmed ego network_repository lr=1e-2 ${seed} train.lr=1e-2 &
# launch pubmed ego network_repository lr=5e-4 ${seed} train.lr=5e-4 &
# launch pubmed ego network_repository lr=1e-4 ${seed} train.lr=1e-4 &

### wdecay
# launch cora ego network_repository wdecay=0 ${seed} train.wdecay=0 &
# launch cora ego network_repository wdecay=1e-9 ${seed} train.wdecay=1e-9 &
# launch cora ego network_repository wdecay=1e-8 ${seed} train.wdecay=1e-8 &
# launch cora ego network_repository wdecay=1e-7 ${seed} train.wdecay=1e-7 &
# launch cora ego network_repository wdecay=1e-6 ${seed} train.wdecay=1e-6 &
# launch cora ego network_repository wdecay=1e-5 ${seed} train.wdecay=1e-5 &

# launch citeseer ego network_repository wdecay=0 ${seed} train.wdecay=0 &
# launch citeseer ego network_repository wdecay=1e-9 ${seed} train.wdecay=1e-9 &
# launch citeseer ego network_repository wdecay=1e-8 ${seed} train.wdecay=1e-8 &
# launch citeseer ego network_repository wdecay=1e-7 ${seed} train.wdecay=1e-7 &
# launch citeseer ego network_repository wdecay=1e-6 ${seed} train.wdecay=1e-6 &
# launch citeseer ego network_repository wdecay=1e-5 ${seed} train.wdecay=1e-5 &

# launch pubmed ego network_repository wdecay=0 ${seed} train.wdecay=0 &
# launch pubmed ego network_repository wdecay=5e-9 ${seed} train.wdecay=5e-9 &
# launch pubmed ego network_repository wdecay=1e-8 ${seed} train.wdecay=1e-8 &
# launch pubmed ego network_repository wdecay=5e-8 ${seed} train.wdecay=5e-8 &
# launch pubmed ego network_repository wdecay=1e-7 ${seed} train.wdecay=1e-7 &
# launch pubmed ego network_repository wdecay=5e-7 ${seed} train.wdecay=5e-7 &
# launch pubmed ego network_repository wdecay=1e-6 ${seed} train.wdecay=1e-6 &

### drop_ratio
# launch cora ego network_repository drop_ratio=0 ${seed} model.drop_ratio=0 &
# launch cora ego network_repository drop_ratio=0.1 ${seed} model.drop_ratio=0.1 &
# launch cora ego network_repository drop_ratio=0.2 ${seed} model.drop_ratio=0.2 &
# launch cora ego network_repository drop_ratio=0.3 ${seed} model.drop_ratio=0.3 &
# launch cora ego network_repository drop_ratio=0.4 ${seed} model.drop_ratio=0.4 &
# launch cora ego network_repository drop_ratio=0.5 ${seed} model.drop_ratio=0.5 &
# launch cora ego network_repository drop_ratio=0.6 ${seed} model.drop_ratio=0.6 &
# launch cora ego network_repository drop_ratio=0.7 ${seed} model.drop_ratio=0.7 &

# launch cora rw network_repository drop_ratio=0 ${seed} model.drop_ratio=0 &
# launch cora rw network_repository drop_ratio=0.1 ${seed} model.drop_ratio=0.1 &
# launch cora rw network_repository drop_ratio=0.2 ${seed} model.drop_ratio=0.2 &
# launch cora rw network_repository drop_ratio=0.3 ${seed} model.drop_ratio=0.3 &
# launch cora rw network_repository drop_ratio=0.4 ${seed} model.drop_ratio=0.4 &
# launch cora rw network_repository drop_ratio=0.5 ${seed} model.drop_ratio=0.5 &
# launch cora rw network_repository drop_ratio=0.5 ${seed} model.drop_ratio=0.6 &

# launch citeseer ego network_repository drop_ratio=0 ${seed} model.drop_ratio=0 &
# launch citeseer ego network_repository drop_ratio=0.1 ${seed} model.drop_ratio=0.1 &
# launch citeseer ego network_repository drop_ratio=0.2 ${seed} model.drop_ratio=0.2 &
# launch citeseer ego network_repository drop_ratio=0.3 ${seed} model.drop_ratio=0.3 &
# launch citeseer ego network_repository drop_ratio=0.4 ${seed} model.drop_ratio=0.4 &
# launch citeseer ego network_repository drop_ratio=0.5 ${seed} model.drop_ratio=0.5 &

# launch pubmed ego network_repository drop_ratio=0 ${seed} model.drop_ratio=0 &
# launch pubmed ego network_repository drop_ratio=0.1 ${seed} model.drop_ratio=0.1 &
# launch pubmed ego network_repository drop_ratio=0.2 ${seed} model.drop_ratio=0.2 &
# launch pubmed ego network_repository drop_ratio=0.3 ${seed} model.drop_ratio=0.3 &
# launch pubmed ego network_repository drop_ratio=0.4 ${seed} model.drop_ratio=0.4 &
# launch pubmed ego network_repository drop_ratio=0.5 ${seed} model.drop_ratio=0.5 &
# launch pubmed ego network_repository drop_ratio=0.6 ${seed} model.drop_ratio=0.6 &
# launch pubmed ego network_repository drop_ratio=0.7 ${seed} model.drop_ratio=0.7 &

### subset
# launch cora ego network_repository subset=True ${seed} model.subset=True &
# launch cora ego network_repository subset=False ${seed} model.subset=False &

# launch cora rw network_repository subset=True ${seed} model.subset=True &
# launch cora rw network_repository subset=False ${seed} model.subset=False &

# launch citeseer ego network_repository subset=True ${seed} model.subset=True &
# launch citeseer ego network_repository subset=False ${seed} model.subset=False &

# launch pubmed ego network_repository subset=True ${seed} model.subset=True &
# launch pubmed ego network_repository subset=False ${seed} model.subset=False &

# launch pubmed rw network_repository subset=True ${seed} model.subset=True &
# launch pubmed rw network_repository subset=False ${seed} model.subset=False &

### predictor_flag
# launch cora ego network_repository predictor_flag=True ${seed} model.predictor_flag=True &
# launch cora ego network_repository predictor_flag=False ${seed} model.predictor_flag=False &

# launch cora rw network_repository predictor_flag=True ${seed} model.predictor_flag=True &
# launch cora rw network_repository predictor_flag=False ${seed} model.predictor_flag=False &

# launch citeseer ego network_repository predictor_flag=True ${seed} model.predictor_flag=True &
# launch citeseer ego network_repository predictor_flag=False ${seed} model.predictor_flag=False &

# launch citeseer rw network_repository predictor_flag=True ${seed} model.predictor_flag=True &
# launch citeseer rw network_repository predictor_flag=False ${seed} model.predictor_flag=False &

# launch pubmed ego network_repository predictor_flag=True ${seed} model.predictor_flag=True &
# launch pubmed ego network_repository predictor_flag=False ${seed} model.predictor_flag=False &

# launch pubmed rw network_repository predictor_flag=True ${seed} model.predictor_flag=True &
# launch pubmed rw network_repository predictor_flag=False ${seed} model.predictor_flag=False &


# augment -------------------------------------------------------------------

### pre_data
# launch cora ego network_repository network_repository ${seed} &
# launch cora ego planetoid planetoid ${seed} &
# launch cora ego cora cora ${seed} &

### out_steps
# launch cora ego network_repository out_steps=1 ${seed} augment.out_steps=1 &
# launch cora ego network_repository out_steps=5 ${seed} augment.out_steps=5 &
# launch cora ego network_repository out_steps=10 ${seed} augment.out_steps=10 &
# launch cora ego network_repository out_steps=50 ${seed} augment.out_steps=50 &
# launch cora ego network_repository out_steps=100 ${seed} augment.out_steps=100 &
# launch cora ego network_repository out_steps=500 ${seed} augment.out_steps=500 &

### topk
# launch cora ego network_repository topk=5 ${seed} augment.topk=5 &
# launch cora ego network_repository topk=10 ${seed} augment.topk=10 &
# launch cora ego network_repository topk=20 ${seed} augment.topk=20 &
# launch cora ego network_repository topk=50 ${seed} augment.topk=50 &
# launch cora ego network_repository topk=128 ${seed} augment.topk=128 &

### n_negative
# launch cora ego network_repository n_negative=1 ${seed} augment.n_negative=1 &
# launch cora ego network_repository n_negative=10 ${seed} augment.n_negative=10 &
# launch cora ego network_repository n_negative=50 ${seed} augment.n_negative=50 &
# launch cora ego network_repository n_negative=100 ${seed} augment.n_negative=100 &

### perturb_ratio
# launch cora ego network_repository perturb_ratio=0.5 ${seed} augment.perturb_ratio=0.5 &
# launch cora ego network_repository perturb_ratio=0.2 ${seed} augment.perturb_ratio=0.2 &
# launch cora ego network_repository perturb_ratio=0.1 ${seed} augment.perturb_ratio=0.1 &
# launch cora ego network_repository perturb_ratio=5e-2 ${seed} augment.perturb_ratio=5e-2 &
# launch cora ego network_repository perturb_ratio=1e-3 ${seed} augment.perturb_ratio=1e-3 &
# launch cora ego network_repository perturb_ratio=5e-4 ${seed} augment.perturb_ratio=5e-4 &

### snr
# launch cora ego network_repository snr=0.1 ${seed} augment.snr=0.1 &
# launch cora ego network_repository snr=0.3 ${seed} augment.snr=0.3 &
# launch cora ego network_repository snr=0.5 ${seed} augment.snr=0.5 &
# launch cora ego network_repository snr=0.7 ${seed} augment.snr=0.7 &
# launch cora ego network_repository snr=0.9 ${seed} augment.snr=0.9 &

# -----------------------------------------------------------------------------


wait