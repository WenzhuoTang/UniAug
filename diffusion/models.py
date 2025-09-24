import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch_scatter import scatter
from torch_geometric.utils import degree
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv, TransformerConv

from diffusion.layers import (
    MLP, EdgeRegressionHead, MiniAttentionLayer,
    create_activation, create_norm, 
    SinusoidalPosEmb, TimeEmb, Mish
)
from diffusion.graph_utils import (
    mask_adjs, pow_tensor,
    KeyPropRandomWalk,
    MessagePropRandomWalk,
)
from utils.models.attention import AttentionLayer
from dataset.loader import MultiEpochsDataLoader


# TODO: implement models in GDSS and DiGress
class GNN(torch.nn.Module):
    def __init__(self, out_channels, emb_type='feat_linear', in_channels=None, max_degree=None, hidden_channels=64, 
                 num_layers=2, gnn_type='gcn', heads=4, dropout=0.5, act='relu', norm=None, agg='concat', num_timesteps=128,
                 final_batch_size=None, num_edge_thres=None, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.F_act = create_activation(act)

        conv_params = {}
        if gnn_type == 'gcn':
            conv_func = GCNConv
        elif gnn_type == 'sage':
            conv_func = SAGEConv
        elif gnn_type == 'gat':
            conv_params['heads'] = heads
            conv_params['concat'] = False
            conv_func = GATConv
        elif gnn_type == 'gt':
            conv_params['heads'] = heads
            conv_params['concat'] = False
            conv_func = TransformerConv
        else:
            raise ValueError(f'Undefined GNN type {gnn_type}')
        
        self.emb_type = emb_type
        if self.emb_type.startswith('feat'):
            assert in_channels is not None
            if self.emb_type.endswith('linear'):
                self.input_trans = nn.Linear(in_channels, hidden_channels)
            else:
                raise NotImplementedError(f'Unsopported embedding type {self.emb_type}')

        elif self.emb_type.startswith('degree'):
            assert max_degree is not None
            self.max_degree = max_degree
            if self.emb_type.endswith('linear'):
                self.input_trans = nn.Linear(1, hidden_channels)
            elif self.emb_type.endswith('embedding'):
                self.input_trans = nn.Embedding(self.max_degree + 1, hidden_channels)
            else:
                raise NotImplementedError(f'Unsopported embedding type {self.emb_type}')

        else:
            raise NotImplementedError(f'Unsopported embedding type {self.emb_type}')

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_channels, num_timesteps),
            nn.Linear(hidden_channels, 4 * hidden_channels),
            nn.SiLU(),
            nn.Linear(4 * hidden_channels, hidden_channels),
        )

        # assert num_layers > 1
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for layer in range(num_layers):
            # if layer == 0:
            #     self.convs.append(conv_func(in_channels, hidden_channels))
            #     self.norms.append(create_norm(norm, hidden_channels))
            # elif layer == num_layers - 1:
            #     self.convs.append(conv_func(hidden_channels, out_channels))
            #     self.norms.append(create_norm(norm, out_channels))
            # else:
            self.convs.append(conv_func(hidden_channels, hidden_channels, **conv_params))
            self.norms.append(create_norm(norm, hidden_channels))

        self.num_edge_thres = num_edge_thres
        self.final_batch_size = final_batch_size
        self.final_out = EdgeRegressionHead(num_layers=2, input_dim=hidden_channels, output_dim=out_channels,
                                  activate_func=F.silu, agg=agg)

    def forward_embedder(self, data, time_steps):
        edge_index = data.edge_index
        # project input
        if self.emb_type.startswith('feat'):
            x = data.x
            x = F.dropout(x, p=self.dropout, training=self.training)

        elif self.emb_type.startswith('degree'):
            x = degree(edge_index[0], num_nodes=data.num_nodes).clamp(max=self.max_degree).long()
            if self.emb_type.endswith('linear'):
                x = x[..., None] / self.max_degree

        h = self.input_trans(x)

        # inject time info
        t_emb = self.time_emb(time_steps)
        h = h + t_emb

        return h
    
    def forward_encoder(self, h, edge_index):
        for i in range(self.num_layers):
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.convs[i](h, edge_index)
            h = self.norms[i](h)
            if i < self.num_layers - 1:
                h = self.F_act(h)

        return h

    def forward_decoder(self, h, src, dst):
        batched_flag = False
        if self.num_edge_thres is not None and self.final_batch_size is not None:
            if len(src) > self.num_edge_thres:
                batched_flag = True
                dataset = TensorDataset(src.cpu(), dst.cpu())
                # dataset = TensorDataset(src, dst)
                dataloader = DataLoader(dataset, batch_size=int(self.final_batch_size), shuffle=False)

                edge_logits_list = []
                for b_src, b_dst in dataloader:
                    b_src, b_dst = b_src.to(h.device), b_dst.to(h.device)
                    edge_logits_list.append(self.final_out(h, b_src, b_dst))
                edge_logits = torch.cat(edge_logits_list)

        if not batched_flag:
            edge_logits = self.final_out(h, src, dst)

        return edge_logits
    
    def get_latent(self, data, time_steps):
        edge_index = data.edge_index

        # project input and inject time info
        h = self.forward_embedder(data, time_steps)
        # h = F.relu(h)

        # graph encoder
        h = self.forward_encoder(h, edge_index)

        return h

    def forward(self, data, time_steps, **kwargs):
        edge_index = data.edge_index
        h_list = []

        # project input and inject time info
        h = self.forward_embedder(data, time_steps)
        h_list.append(h)
        # h = F.relu(h)

        # graph encoder
        h = self.forward_encoder(h, edge_index)
        h_list.append(h)
        # h = self.F_act(h)

        # edge decoder
        src, dst = data.full_edge_index[0],  data.full_edge_index[1]
        edge_logits = self.forward_decoder(h, src, dst)

        return edge_logits, h_list


class GuidedGNN(GNN):
    def __init__(self, out_channels, num_classes, emb_type='feat_linear', in_channels=None, max_degrees=None, hidden_channels=64, 
                 num_layers=2, gnn_type='gcn', heads=4, dropout=0.5, act='relu', norm=None, agg='concat', num_timesteps=128,
                 final_batch_size=None, num_edge_thres=None, **kwargs):
        super().__init__(out_channels, emb_type=emb_type, in_channels=in_channels, max_degree=max_degrees[0], 
                         hidden_channels=hidden_channels, num_layers=num_layers, gnn_type=gnn_type, heads=heads, 
                         dropout=dropout, act=act, norm=norm, agg=agg, num_timesteps=num_timesteps,
                         final_batch_size=final_batch_size, num_edge_thres=num_edge_thres)
        
        # overwrite embedders
        # self.classifier = classifier  # sklearn kmeans
        # self.num_classes = self.classifier.n_clusters
        self.num_classes = num_classes
        self.class_emb = nn.Embedding(self.num_classes, hidden_channels)

        self.emb_type = emb_type
        self.input_trans = torch.nn.ModuleList()

        # one embedder per class
        for i in range(self.num_classes):
            if self.emb_type.startswith('feat'):
                assert in_channels is not None
                if self.emb_type.endswith('linear'):
                    self.input_trans.append(nn.Linear(in_channels, hidden_channels))
                else:
                    raise NotImplementedError(f'Unsopported embedding type {self.emb_type}')

            elif self.emb_type.startswith('degree'):
                assert max_degrees is not None  # max_degrees should be a dict
                self.max_degrees = torch.tensor(max_degrees)
                if self.emb_type.endswith('linear'):
                    self.input_trans.append(nn.Linear(1, hidden_channels))
                elif self.emb_type.endswith('embedding'):
                    self.input_trans.append(nn.Embedding(self.max_degrees[i] + 1, hidden_channels))
                else:
                    raise NotImplementedError(f'Unsopported embedding type {self.emb_type}')

            else:
                raise NotImplementedError(f'Unsopported embedding type {self.emb_type}')

    # overwrite embedders
    def forward_embedder(self, data, time_steps):
        c_pre, edge_index = data.kmeans_labels, data.edge_index

        # # predict class
        # class_feat = data.property_attr
        # c_pre = torch.from_numpy(self.classifier.predict(class_feat.cpu())).to(edge_index.device)
        # c_pre = c_pre.repeat_interleave(data.nodes_per_graph)

        # project input
        if self.emb_type.startswith('feat'):
            x = data.x
            x = F.dropout(x, p=self.dropout, training=self.training)

        elif self.emb_type.startswith('degree'):
            self.max_degrees = self.max_degrees.to(edge_index.device)
            x = degree(edge_index[0], num_nodes=data.num_nodes)
            x = x.clamp(max=self.max_degrees[c_pre]).long()
            if self.emb_type.endswith('linear'):
                x = (x / self.max_degrees[c_pre]).reshape(-1, 1)

        order = []
        hlist = []
        for i in c_pre.unique():
            idx = torch.where(c_pre == i)[0]
            order.append(idx)
            hlist.append(self.input_trans[i](x[idx]))

        h = torch.cat(hlist)[torch.cat(order)]

        # inject time info
        t_emb = self.time_emb(time_steps)
        h = h + t_emb

        # inject class info
        c_emb = self.class_emb(c_pre)
        h = h + c_emb

        return h


class MSTAGNN(torch.nn.Module):
    """ https://github.com/LUMIA-Group/SubTree-Attention/blob/publish/stagnn.py """

    def __init__(self, in_channels, out_channels, hidden_channels=64, dropout=0., K=3, num_heads=1, ind_gamma=True, 
                 gamma_softmax=True, multi_concat=True, global_attn=False, agg='concat', num_timesteps=128, 
                 final_batch_size=None, **kwargs):
        super(MSTAGNN, self).__init__()
        self.headc = headc = hidden_channels // num_heads
        self.dim_v = headc
        self.input_trans = nn.Linear(in_channels, hidden_channels)
        self.linQ = nn.Linear(hidden_channels, headc * num_heads)
        self.linK = nn.Linear(hidden_channels, headc * num_heads)
        self.linV = nn.Linear(hidden_channels, self.dim_v * num_heads)
        if (multi_concat):
            self.output = nn.Linear(self.dim_v * num_heads, self.dim_v)

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_channels, num_timesteps),
            nn.Linear(hidden_channels, 4 * hidden_channels),
            nn.SiLU(),
            nn.Linear(4 * hidden_channels, hidden_channels),
        )

        self.propM = MessagePropRandomWalk(node_dim=-4)
        self.propK = KeyPropRandomWalk(node_dim=-3)        

        self.dropout = dropout
        self.K = K
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.multi_concat = multi_concat
        self.ind_gamma = ind_gamma
        self.gamma_softmax = gamma_softmax
        self.global_attn = global_attn

        self.cst = 10e-6
        if (ind_gamma):
            if (gamma_softmax):
                self.hopwise = nn.Parameter(torch.ones(K+1))
                self.headwise = nn.Parameter(torch.zeros(size=(self.num_heads,K)))
            else:
                self.hopwise = nn.Parameter(torch.ones(size=(self.num_heads,K+1)))
        else:
            self.hopwise = nn.Parameter(torch.ones(K+1))
        
        self.teleport = nn.Parameter(torch.ones(1))

        self.final_batch_size = final_batch_size
        self.final_out = EdgeRegressionHead(num_layers=2, input_dim=self.dim_v, output_dim=out_channels,
                                  activate_func=F.silu, agg=agg)

    def reset_parameters(self):
        if (self.ind_gamma and self.gamma_softmax):
            torch.nn.init.ones_(self.hopwise)
            torch.nn.init.zeros_(self.headwise)
        else:
            torch.nn.init.ones_(self.hopwise)
        self.input_trans.reset_parameters()
        self.linQ.reset_parameters()
        self.linK.reset_parameters()
        self.linV.reset_parameters()
        if (self.multi_concat):
            self.output.reset_parameters()
        torch.nn.init.ones_(self.teleport)

    def forward(self, data, time_steps, final_batch_size=None):
        x, edge_index = data.x, data.edge_index

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row]

        # project input and inject time info
        t_emb = self.time_emb(time_steps)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.input_trans(x) + t_emb)
        x = F.dropout(x, p=self.dropout, training=self.training)

        Q = self.linQ(x)
        K = self.linK(x)
        V = self.linV(x)

        Q = 1 + F.elu(Q)
        K = 1 + F.elu(K)

        Q = Q.view(-1, self.num_heads, self.headc)
        K = K.view(-1, self.num_heads, self.headc)
        V = V.view(-1, self.num_heads, self.dim_v)

        M = torch.einsum('nhi,nhj->nhij', [K, V])

        if (self.ind_gamma):
            if (self.gamma_softmax):
                hidden = V * (self.hopwise[0])
            else:
                hidden = V * (self.hopwise[:, 0].unsqueeze(-1))
        else:
            hidden = V * (self.hopwise[0])

        if ((self.ind_gamma) and (self.gamma_softmax)):
            layerwise = F.softmax(self.headwise, dim=-2)

        if (self.global_attn):
            num_nodes = x.size(0)
            teleportM = torch.sum(M, dim=0, keepdim=True) / num_nodes
            teleportK = torch.sum(K, dim=0, keepdim=True) / num_nodes
            teleportH = torch.einsum('nhi,nhij->nhj',[Q,teleportM])
            teleportC = torch.einsum('nhi,nhi->nh',[Q,teleportK]).unsqueeze(-1) + self.cst
            teleportH = teleportH / teleportC
            teleportH = teleportH.sum(dim=-2)

        for hop in range(self.K):
            M = self.propM(M, edge_index, norm.view(-1,1,1,1))
            K = self.propK(K, edge_index, norm.view(-1,1,1))

            H = torch.einsum('nhi,nhij->nhj', [Q, M])
            C = torch.einsum('nhi,nhi->nh', [Q, K]).unsqueeze(-1) + self.cst
            H = H / C
            if (self.ind_gamma):
                if (self.gamma_softmax):
                    gamma = self.hopwise[hop+1] * layerwise[:, hop].unsqueeze(-1)
                else:
                    gamma = self.hopwise[:, hop+1].unsqueeze(-1)
            else:
                gamma = self.hopwise[hop+1]
            hidden = hidden + gamma * H

        if (self.multi_concat):
            hidden = hidden.view(-1, self.dim_v * self.num_heads)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            hidden = self.output(hidden)
        else:
            hidden = hidden.sum(dim=-2)
    
        if (self.global_attn):
            hidden = hidden + self.teleport*teleportH
        
        # edge decoder
        if final_batch_size is not None:
            self.final_batch_size = final_batch_size

        if self.final_batch_size is not None:
            device = data.full_edge_index.device
            dataset = TensorDataset(data.full_edge_index.T.cpu())
            dataloader = MultiEpochsDataLoader(dataset, batch_size=self.final_batch_size, 
                                               shuffle=False, num_workers=16)
            edge_logits_list = []
            for b_index in dataloader:
                b_index = b_index[0].to(device)
                src, dst = b_index.T
                edge_logits_list.append(self.final_out(hidden, src, dst))
            edge_logits = torch.cat(edge_logits_list)

        else:
            src, dst = data.full_edge_index[0],  data.full_edge_index[1]
            edge_logits = self.final_out(hidden, src, dst)

        # row, col = data.full_edge_index[0],  data.full_edge_index[1]
        # edge_logits = self.final_out(hidden, row, col)
        return edge_logits, [hidden]


class TGNN(torch.nn.Module):
    def __init__(self, max_degree, out_channels, dim, num_timesteps=128, num_heads=4, num_layers=4, dropout=0., 
                 norm=None, agg='concat', final_batch_size=None, **kwargs):
        super().__init__()
        self.max_degree = max_degree
        self.out_channels = out_channels
        num_heads = [num_heads] * (num_layers - 1) + [1]
        self.num_heads = num_heads 
        self.dim = dim
        self.num_timesteps = num_timesteps
        self.embedding_t = torch.nn.Linear(1, dim)
        self.time_pos_emb = SinusoidalPosEmb(dim, num_timesteps)
        self.layers = torch.nn.ModuleDict()
        self.norm = norm
        self.gru = torch.nn.Identity()
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.context_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )  

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.dropout = torch.nn.Dropout(p=dropout)
        self.gru = torch.nn.GRU(dim, dim)

        for i, num_head in enumerate(num_heads):
            self.layers[f'time{i}'] = TimeEmb(dim, dim, Mish())
            self.layers[f'conv{i}'] = TransformerConv(in_channels=dim*2, out_channels=dim, heads=num_head, concat=False)
            self.layers[f'norm{i}'] = create_norm(norm, dim)
            self.layers[f'act{i}'] = torch.nn.SiLU()

        self.dummy_edge_feats = torch.nn.parameter.Parameter(torch.randn(dim))
        self.node_interaction = MiniAttentionLayer(node_dim=dim, in_edge_dim=dim, out_edge_dim=dim, d_model=dim, num_heads=2)
        
        self.final_batch_size = final_batch_size
        self.final_out = EdgeRegressionHead(num_layers=2, input_dim=dim, output_dim=out_channels,
                                  activate_func=F.silu, agg=agg)
        
    def forward(self, pyg_data, t_node, final_batch_size=None):
        edge_index = pyg_data.edge_index
        nodes = degree(edge_index[0], num_nodes=pyg_data.num_nodes).clamp(max=self.max_degree+1).long()

        nodes = nodes[..., None] / self.max_degree  # I prefer to make it embedding later
        nodes = self.embedding_t(nodes)
        t = self.time_pos_emb(t_node)
        t = self.mlp(t)
        
        h = nodes.unsqueeze(0)
        contexts = scatter(nodes, pyg_data.batch, reduce='mean', dim=0)
        contexts = self.global_mlp(contexts)

        contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)

        for i in range(len(self.num_heads)):
            ### add time embedding ###
            t_emb = self.layers[f'time{i}'](t)

            nodes = torch.cat([nodes, t_emb], dim=-1)
            
            ### message passing on graph ###
            nodes = self.layers[f'conv{i}'](nodes, edge_index)
            nodes = self.layers[f'norm{i}'](nodes)
            nodes = self.layers[f'act{i}'](nodes)
            nodes = self.dropout(nodes)

            ### gru update ###
            nodes, h = self.gru(nodes.unsqueeze(0).contiguous(), h.contiguous())
            h = self.dropout(h)
            nodes = nodes.squeeze(0)
            
            ### global context aggregation ###
            # aggregate locals to global
            node_contexts = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts = scatter(contexts + node_contexts, pyg_data.batch, reduce='mean', dim=0)
            contexts = self.global_mlp(contexts)
            contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)
            # spread global to locals
            nodes = nodes + contexts

        # edge decoder
        if final_batch_size is not None:
            self.final_batch_size = final_batch_size

        if self.final_batch_size is not None:
            device = pyg_data.full_edge_index.device
            dataset = TensorDataset(pyg_data.full_edge_index.T.cpu())
            dataloader = MultiEpochsDataLoader(dataset, batch_size=self.final_batch_size, 
                                               shuffle=False, num_workers=16)
            edge_logits_list = []
            for b_index in dataloader:
                b_index = b_index[0].to(device)
                src, dst = b_index.T
                edge_logits_list.append(self.final_out(nodes, src, dst))
            edge_logits = torch.cat(edge_logits_list)

        else:
            src, dst = pyg_data.full_edge_index[0],  pyg_data.full_edge_index[1]
            edge_logits = self.final_out(nodes, src, dst)

        # row, col = pyg_data.full_edge_index[0],  pyg_data.full_edge_index[1]
        # edge_logits = self.final_out(nodes, row, col)

        return edge_logits, [nodes]


class TGNNDegreeGuided(torch.nn.Module):
    def __init__(self, max_degree, out_channels, dim, num_timesteps, num_heads=[4, 4, 4, 1], dropout=0., 
                 norm=None, agg='sum', **kwargs):
        super().__init__()
        self.max_degree = max_degree
        self.out_channels = out_channels
        self.num_heads = num_heads 
        self.dim = dim
        self.num_timesteps = num_timesteps
        self.embedding_t = torch.nn.Linear(1, dim)
        self.embedding_0 = torch.nn.Linear(1, dim)
        self.embedding_sel = torch.nn.Embedding(2, dim)
        self.node_in = torch.torch.nn.Sequential(
            torch.nn.Linear(dim * 3, dim),
            torch.nn.SiLU()
        )
        self.time_pos_emb = SinusoidalPosEmb(dim, num_timesteps)
        self.layers = torch.nn.ModuleDict()
        self.norm = norm
        self.gru = torch.nn.Identity()
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.context_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim*2, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )  

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 4, dim)
            )

        self.dropout = torch.nn.Dropout(p=dropout)
        self.gru = torch.nn.GRU(dim, dim)

        for i, num_head in enumerate(num_heads):
            self.layers[f'time{i}'] = TimeEmb(dim, dim, Mish())
            self.layers[f'conv{i}'] = TransformerConv(in_channels=dim*2, out_channels=dim, heads=num_head, concat=False)
            self.layers[f'norm{i}'] = create_norm(norm, dim)
            self.layers[f'act{i}'] = torch.nn.SiLU()

        self.dummy_edge_feats = torch.nn.parameter.Parameter(torch.randn(dim))

        self.node_out_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim * 4, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(dim * 2, dim)
        )

        self.final_out = EdgeRegressionHead(num_layers=2, input_dim=dim, output_dim=out_channels,
                                  activate_func=F.silu, agg=agg)
        
    def forward(self, pyg_data, t_node):
        edge_index = pyg_data.edge_index
        nodes_t = degree(edge_index[0],num_nodes=pyg_data.num_nodes).clamp(max=self.max_degree+1).long()
        node_selection = torch.zeros_like(nodes_t)


        nodes_t = nodes_t[..., None] / self.max_degree  # I prefer to make it embedding later
        nodes_0 = pyg_data.degree[..., None] / self.max_degree
        active_node_indices = (nodes_0 != nodes_t).nonzero(as_tuple=True)[0]
        node_selection[active_node_indices] = 1
        node_selection = node_selection.long()
        
        nodes = torch.cat([self.embedding_t(nodes_t), self.embedding_0(nodes_0), self.embedding_sel(node_selection)], dim=-1)
        nodes = self.node_in(nodes)

        t = self.time_pos_emb(t_node)
        t = self.mlp(t)
        
        h = nodes.unsqueeze(0)
        contexts = scatter(nodes, pyg_data.batch, reduce='mean', dim=0)
        contexts = self.global_mlp(contexts)

        contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)

        for i in range(len(self.num_heads)):
            ### add time embedding ###
            t_emb = self.layers[f'time{i}'](t)

            nodes = torch.cat([nodes, t_emb], dim=-1)
            
            ### message passing on graph ###
            nodes = self.layers[f'conv{i}'](nodes, edge_index)
            nodes = self.layers[f'norm{i}'](nodes)
            nodes = self.layers[f'act{i}'](nodes)
            nodes = self.dropout(nodes)

            ### gru update ###
            nodes, h = self.gru(nodes.unsqueeze(0).contiguous(), h.contiguous())
            h = self.dropout(h)
            nodes = nodes.squeeze(0)
            
            ### global context aggregation ###
            # aggregate locals to global
            node_contexts = self.context_mlp(torch.cat([nodes, contexts], dim=-1))
            contexts = scatter(contexts + node_contexts, pyg_data.batch, reduce='mean', dim=0)
            contexts = self.global_mlp(contexts)
            contexts = contexts.repeat_interleave(pyg_data.nodes_per_graph,dim=0)
            # spread global to locals
            nodes = nodes + contexts

        # mlp add
        # row = pyg_data.full_edge_index[0].index_select(0, pyg_data.active_edge_indices)
        # col = pyg_data.full_edge_index[1].index_select(0, pyg_data.active_edge_indices)
        row, col = pyg_data.full_edge_index[0],  pyg_data.full_edge_index[1]

        nodes = torch.cat([nodes, self.embedding_t(nodes_t), self.embedding_0(nodes_0), self.embedding_sel(node_selection)], dim=-1)
        nodes = self.node_out_mlp(nodes)

        edge_logits = self.final_out(nodes, row, col)
        
        return edge_logits, [nodes]


class PowerGraphTransformer(torch.nn.Module):
    """ Modified from GDSS """

    def __init__(self, max_feat_num, nhid=32, num_layers=3, num_linears=2, c_init=2, c_hid=8, c_final=4, 
                 adim=32, num_heads=4, conv='GCN', make_symetric=False):

        super().__init__()
        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv
        self.num_layers = num_layers
        self.make_symetric = make_symetric

        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            if i==0:
                self.layers.append(AttentionLayer(num_linears, max_feat_num, nhid, nhid, c_init, 
                                                  c_hid, num_heads, conv))
            elif i==self.num_layers-1:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
                                                  c_final, num_heads, conv))
            else:
                self.layers.append(AttentionLayer(num_linears, nhid, adim, nhid, c_hid, 
                                                  c_hid, num_heads, conv))

        fdim = c_hid * (num_layers - 1) + c_final + c_init
        self.final = MLP(num_layers=3, input_dim=fdim, hidden_dim=2 * fdim, output_dim=1, 
                         use_bn=False, activate_func=F.silu)

    def forward(self, x, adj, flags):
        adjc = pow_tensor(adj, self.c_init)
        adj_list = [adjc]
        for i in range(self.num_layers):
            x, adjc = self.layers[i](x, adjc, flags)
            adj_list.append(adjc)
        
        adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
        out_shape = adjs.shape[:-1] # B x N x N
        adj_out = self.final(adjs).view(*out_shape)
        if self.make_symetric:
            adj_out = 1/2 * (adj_out + torch.transpose(adj_out, 1, 2))

        mask = torch.eye(adj_out.size(-1), adj_out.size(-1)).bool().to(adj_out.device).unsqueeze_(0) 
        adj_out.masked_fill_(mask, 0)
        adj_out = mask_adjs(adj_out, flags)

        return adj_out