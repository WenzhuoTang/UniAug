import argparse

class Parser:
    """ parse class for pretrain GDSS"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='GDSS')
        # self.parser.add_argument('--type', type=str, required=True)
        self.set_arguments()
    
    def str2bool(self, v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    
    def none_or_float(self, value):
        if value == 'None':
            return None
        return float(value)

    def set_arguments(self):
        self.parser.add_argument('--config', type=str,
                                    required=True, help="Path of config file")
        self.parser.add_argument('--comment', type=str, default="", 
                                    help="A single line comment for the experiment")
        self.parser.add_argument('--prefix', type=str, default="", 
                                    help="Prefix of checkpoint name or run name")
        self.parser.add_argument('--seed', type=int, default=1)
        self.parser.add_argument('--no_print', action='store_true', default=False,
                            help="don't use progress bar")
        self.parser.add_argument('--num_workers', type=int, default=0,
                            help='number of workers for data loader')
        self.parser.add_argument('--augment', type=self.str2bool, default=False,
                            help='perform augmentation or not')
        self.parser.add_argument('--sweep', type=self.str2bool, default=False,
                            help='flag of conducting wandb sweep')
        self.parser.add_argument('--sweep_id', type=str, default=None,
                            help='wandb sweep id')
        self.parser.add_argument('--profile', type=self.str2bool, default=False,
                            help='flag of pytorch profile')
        self.parser.add_argument('--orig_feat', type=self.str2bool, default=False,
                            help='flag of running on original features')
        self.parser.add_argument('--full_subgraph', type=self.str2bool, default=False,
                            help='flag of subsample nodes or not')
        self.parser.add_argument('--aug_orig', type=self.str2bool, default=False,
                            help='flag of augment original feature or not')
        self.parser.add_argument('--use_val_edges', type=self.str2bool, default=False,
                            help='flag of use val edges for test or not')
        self.parser.add_argument('--remove_dup', type=self.str2bool, default=False,
                            help='flag of remove duplicate edges or not')
        self.parser.add_argument('--train_guidance', type=self.str2bool, default=False,
                            help='flag of train guidance head or not')

        self.parser.add_argument('--fold', type=int, default=None,
                            help='index of CV fold')

        self.parser.add_argument('--save_prediction', type=self.str2bool, default=False,
                            help='save node class prediction')
        self.parser.add_argument('--predict_all', type=self.str2bool, default=False,
                            help='whether to predict train and validation labels')
        
        self.parser.add_argument('--neg_guide', type=self.str2bool, default=False,
                            help='train guidance on other dataset or not')
        
        self.parser.add_argument('--halfhop', type=self.str2bool, default=False,
                            help='apply halfhop augmentation or not')
        self.parser.add_argument('--alpha', type=float, default=0.5,
                            help='hyperparameter of halfhop')
        self.parser.add_argument('--p', type=float, default=1.0,
                            help='hyperparameter of halfhop')

        # augmentation
        self.parser.add_argument('--thres', type=self.none_or_float, default=None)
        
        # task & data
        self.parser.add_argument('--pre_data_name', default="all", type=str,
                            choices=['cora', 'citeseer', 'pubmed', 'all'], help='pretrain dataset name')
        self.parser.add_argument('--pre_task', default="both", type=str,
                            choices=['LP', 'NC', 'both'], help='pretrain task name')
        self.parser.add_argument('--data_name', default="cora", type=str,
                            choices=['cora', 'citeseer', 'pubmed', 'all'], help='dataset name')
        self.parser.add_argument('--task', default="NC", type=str,
                            choices=['LP', 'NC', 'both'], help='task name')
        
        # model
        self.parser.add_argument('--model', type=str, default='gin-virtual',
                            help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
        self.parser.add_argument('--readout', type=str, default='sum',
                            help='graph readout (default: sum)')
        self.parser.add_argument('--norm_layer', type=str, default='batch_norm', 
                            help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
        self.parser.add_argument('--drop_ratio', type=float, default=0.5,
                            help='dropout ratio (default: 0.5)')
        self.parser.add_argument('--num_layer', type=int, default=5,
                            help='number of GNN message passing layers (default: 5)')
        self.parser.add_argument('--emb_dim', type=int, default=300,
                            help='dimensionality of hidden units in GNNs (default: 300)')

        # training
        self.parser.add_argument('--batch_size', type=int, default=512,
                            help='input batch size for training (default: 256)')
        self.parser.add_argument('--patience', type=int, default=50,
                            help='patience for early stop')
        self.parser.add_argument('--trails', type=int, default=0,
                            help='nubmer of experiments (default: 5)')   
        self.parser.add_argument('--lr', '--learning_rate', type=float, default=1e-2,
                            help='Learning rate (default: 1e-2)')
        self.parser.add_argument('--wdecay', default=1e-5, type=float,
                            help='weight decay')
        self.parser.add_argument('--epochs', type=int, default=300,
                            help='number of epochs to train')
        self.parser.add_argument('--initw_name', type=str, default='default',
                            help="method to initialize the model paramter")

        # augmentation
        self.parser.add_argument('--ckpt_path', type=str, default='checkpoints/both/all') # put it to config
        self.parser.add_argument('--start', type=int, default=20,
                            help="start epoch for augmentation")
        self.parser.add_argument('--iteration', type=int, default=20,
                            help='epoch to do augmentation')
        self.parser.add_argument('--strategy', default="replace_accumulate", type=str,
                            choices=['replace_once', 'add_once', 'replace_accumulate', 'add_accumulate'],
                            help='  strategy about how to use the augmented examples. \
                                    Replace or add to the original examples; Accumulate the augmented examples or not')
        self.parser.add_argument('--n_jobs', type=int, default=22,
                            help='# process to convert the dense adj input to pyg input form')
        self.parser.add_argument('--n_negative', type=int, default=5,
                            help='# negative samples to optimize the augmented example')
        self.parser.add_argument('--out_steps', type=int, default=5,
                            help='outer sampling steps for guided reverse diffusion')
        self.parser.add_argument('--topk', type=int, default=100,
                            help='top k in an augmentation batch ')
        self.parser.add_argument('--aug_batch', type=int, default=400,
                            help='the augmentation batch compared to training batch')
        self.parser.add_argument('--snr', type=float, default=0.2,
                            help='snr')
        self.parser.add_argument('--scale_eps', type=float, default=0,
                            help='scale eps')
        self.parser.add_argument('--perturb_ratio', type=float, default=None,
                            help='level of noise for perturbation')
        self.parser.add_argument('--n_steps', type=int, default=1,
                            help='n_steps for solver')
        
    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        return args, unparsed

# def get_args():
#     """get args for downstream task"""
#     parser = argparse.ArgumentParser(description='Data-Centric Learning from Unlabeled Graphs with Diffusion Model')
#     parser.add_argument('--config', type=str,
#                         required=True, help="Path of config file")
#     parser.add_argument('--seed', type=int, default=1)
#     parser.add_argument('--gpu_id', type=int, default=0,
#                         help='which gpu to use if any (default: 0)')
#     parser.add_argument('--num_workers', type=int, default=0,
#                         help='number of workers for data loader')
#     parser.add_argument('--no_print', action='store_true', default=False,
#                         help="don't use progress bar")
    
#     # task & data
#     parser.add_argument('--pre_data_name', default="all", type=str,
#                         choices=['cora', 'citeseer', 'pubmed', 'all'], help='pretrain dataset name')
#     parser.add_argument('--pre_task', default="both", type=str,
#                         choices=['LP', 'NC', 'both'], help='pretrain task name')
#     parser.add_argument('--data_name', default="cora", type=str,
#                         choices=['cora', 'citeseer', 'pubmed', 'all'], help='dataset name')
#     parser.add_argument('--task', default="NC", type=str,
#                         choices=['LP', 'NC', 'both'], help='task name')
    
#     # model
#     parser.add_argument('--model', type=str, default='gin-virtual',
#                         help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
#     parser.add_argument('--readout', type=str, default='sum',
#                         help='graph readout (default: sum)')
#     parser.add_argument('--norm_layer', type=str, default='batch_norm', 
#                         help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
#     parser.add_argument('--drop_ratio', type=float, default=0.5,
#                         help='dropout ratio (default: 0.5)')
#     parser.add_argument('--num_layer', type=int, default=5,
#                         help='number of GNN message passing layers (default: 5)')
#     parser.add_argument('--emb_dim', type=int, default=300,
#                         help='dimensionality of hidden units in GNNs (default: 300)')

#     # training
#     parser.add_argument('--batch_size', type=int, default=512,
#                         help='input batch size for training (default: 256)')
#     parser.add_argument('--patience', type=int, default=50,
#                         help='patience for early stop')
#     parser.add_argument('--trails', type=int, default=5,
#                         help='nubmer of experiments (default: 5)')   
#     parser.add_argument('--lr', '--learning_rate', type=float, default=1e-2,
#                         help='Learning rate (default: 1e-2)')
#     parser.add_argument('--wdecay', default=1e-5, type=float,
#                         help='weight decay')
#     parser.add_argument('--epochs', type=int, default=300,
#                         help='number of epochs to train')
#     parser.add_argument('--initw_name', type=str, default='default',
#                         help="method to initialize the model paramter")

#     # augmentation
#     parser.add_argument('--start', type=int, default=20,
#                         help="start epoch for augmentation")
#     parser.add_argument('--iteration', type=int, default=20,
#                         help='epoch to do augmentation')
#     parser.add_argument('--strategy', default="replace_accumulate", type=str,
#                         choices=['replace_once', 'add_once', 'replace_accumulate', 'add_accumulate'],
#                         help='  strategy about how to use the augmented examples. \
#                                 Replace or add to the original examples; Accumulate the augmented examples or not')
#     parser.add_argument('--n_jobs', type=int, default=22,
#                         help='# process to convert the dense adj input to pyg input form')
#     parser.add_argument('--n_negative', type=int, default=5,
#                         help='# negative samples to optimize the augmented example')
#     parser.add_argument('--out_steps', type=int, default=5,
#                         help='outer sampling steps for guided reverse diffusion')
#     parser.add_argument('--topk', type=int, default=100,
#                         help='top k in an augmentation batch ')
#     parser.add_argument('--aug_batch', type=int, default=400,
#                         help='the augmentation batch compared to training batch')
#     parser.add_argument('--snr', type=float, default=0.2,
#                         help='snr')
#     parser.add_argument('--scale_eps', type=float, default=0,
#                         help='scale eps')
#     parser.add_argument('--perturb_ratio', type=float, default=None,
#                         help='level of noise for perturbation')
#     args = parser.parse_args()
#     print('no print',args.no_print)

#     ## n_steps for solver
#     args.n_steps = 1
#     return args