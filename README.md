# UniAug

This is the official implementation for [Cross-Domain Graph Data Scaling: A Showcase with Diffusion Models](https://arxiv.org/pdf/2406.01899), which is a diffusion-based graph data argumentation, with consistent performance gain on node classification, link prediction, and graph classification tasks.


## Abstract

 Models for natural language and images benefit from data scaling behavior: the more data fed into the model, the better they perform. This ’better with more’ phenomenon enables the effectiveness of large-scale pre-training on vast amounts of data. However, current graph pre-training methods struggle to scale up data due to heterogeneity across graphs. To achieve effective data scaling, we aim to develop a general model that is able to capture diverse data patterns of graphs and can be utilized to adaptively help the downstream tasks. To this end, we propose UniAug, a universal graph structure augmentor built on a diffusion model. We f irst pre-train a discrete diffusion model on thousands of graphs across domains to learn the graph structural patterns. In the downstream phase, we provide adaptive enhancement by conducting graph structure augmentation with the help of the pre-trained diffusion model via guided generation. By leveraging the pre-trained diffusion model for structure augmentation, we consistently achieve performance improvements across various downstream tasks in a plug-and-play manner. To the best of our knowledge, this study represents the first demonstration of a data-scaling graph structure augmentor on graphs across domains.


## Enviroment Requirements.

```bash
conda create -f graph_enviroment.yaml
conda activate graph
pip install -r requirements.txt
pip install -f https://data.pyg.org/whl/torch-2.1.0+cu121.html torch-cluster==1.6.3+pt21cu121 torch-scatter==2.1.2+pt21cu121 torch-sparse==0.6.18+pt21cu121 torch-spline-conv==1.2.2+pt21cu121
pip install git+https://github.com/fabriziocosta/EDeN.git --user

```

After executing the shell commands above, there should be a conda enviroment named graph with all dependencies.


## Running Experiments.

All of our shell scripts are under the `run` folder to ease usage.
If you are using HPC,you may need to revise the Resource Request part for scripts with the prefix `hpcc` correspondingly.

Kindly note that for the very first time you may need to login wandb before the execution via `wandb login` following the prompts.

### Graph Prediction

1. To excute the Graph Prediction experiments, two arguments `--config` and `--seed` are required. The config files are under folder `config/graph_pred*`.
2. The `extra_config` are used to config the augment and model detailed settings.
    
    a. pre-trained model can be loaded via the `augment.chpt_path=${chpt_path}`

3. The `extra_args` are used to config the following settings

    a. `--fold` for datasets using K-fold split

    b. `--prefix` for setting the prefix for storing file_path

    c. `--train_guidance`. `--neg_guide`. `--augment`, default=False
```
python3 graph_pred_mol.py --config ${config_file} --seed ${seed} ${extra_args} ${extra_config}
```

For more details, you could check more details in our shell scripts
You could uncomment the corresponding linesin the scripts for different datasets and guidence settings.
To execute our shell scripts, you could use the following command.

```shell
bash run/graph_pred_mol.sh
```

### Node Classification

To execute our shell scripts, you could use the following command.
You could uncomment the corresponding linesin the scripts for different datasets and guidence settings.

```shell
bash run/node_class_subgraph.sh 
```


### Link Prediction

To execute our shell scripts, you could use the following command.
You could uncomment the corresponding lines for different datasets and guidence settings.
```shell
bash bash run/link_pred_ncn.sh
```
