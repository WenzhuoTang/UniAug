import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv,
    global_mean_pool, global_add_pool, global_max_pool
)

from models.utils import create_activation, create_norm
from models.layers import PoolingLayer, convdict, GINNodeVirtualnode
from models.link_pred import DropAdj


# -------- works for node classification --------
class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data, input_hidden=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if not input_hidden:
            x = F.dropout(x, p=self.dropout, training=self.training)
            if edge_attr is not None:
                x = self.conv1(x, edge_index, edge_attr)
            else:
                x = self.conv1(x, edge_index)
        x_hidden = x.detach().clone()
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        if edge_attr is not None:
            x = self.conv2(x, edge_index, edge_attr)
        else:
            x = self.conv2(x, edge_index)
        return x, x_hidden


# -------- works for link prediction --------
class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2, gnn_type='gcn', 
                 heads=4, dropout=0.5, act='relu', norm=None, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.F_act = create_activation(act)

        if gnn_type == 'gcn':
            conv_func = GCNConv
        elif gnn_type == 'sage':
            conv_func = SAGEConv
        elif gnn_type == 'gat':
            conv_hidden= int(hidden_channels / heads)
            conv_out = int(out_channels / heads)
            conv_func = GATConv
        else:
            raise ValueError(f'Undefined GNN type {gnn_type}')

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        if num_layers == 1:
            if gnn_type == 'gat':
                self.convs.append(conv_func(in_channels, conv_out, heads=heads))
            else:
                self.convs.append(conv_func(in_channels, out_channels))
            self.norms.append(create_norm(norm, out_channels))

        elif num_layers > 1:
            for layer in range(num_layers):
                if layer == 0:
                    if gnn_type == 'gat':
                        self.convs.append(conv_func(in_channels, conv_hidden, heads=heads))
                    else:
                        self.convs.append(conv_func(in_channels, hidden_channels))
                    self.norms.append(create_norm(norm, hidden_channels))
                elif layer == num_layers - 1:
                    if gnn_type == 'gat':
                        self.convs.append(conv_func(conv_hidden, conv_out, heads=heads))
                    else:
                        self.convs.append(conv_func(hidden_channels, out_channels))
                    self.norms.append(create_norm(norm, out_channels))
                else:
                    if gnn_type == 'gat':
                        self.convs.append(conv_func(conv_hidden, conv_hidden, heads=heads))
                    else:
                        self.convs.append(conv_func(hidden_channels, hidden_channels))
                    self.norms.append(create_norm(norm, out_channels))

    def forward(self, data=None, x=None, adj=None):
        if data is not None:
            x, edges = data.x, data.edge_index
        else:
            edges = adj
        for i in range(self.num_layers):
            x = self.convs[i](x, edges)
            if i < self.num_layers - 1:
                x = self.norms[i](x)
                x = self.F_act(x)
                x = F.dropout(x, self.dropout, training=self.training)

        return x


# -------- GCN from NCN --------
class NCNGCN(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 ln=False,
                 res=False,
                 max_x=-1,
                 conv_fn="gcn",
                 jk=False,
                 edrop=0.0,
                 xdropout=0.0,
                 taildropout=0.0,
                 noinputlin=False,
                 **kwargs):

        super().__init__()
        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.adjdrop = DropAdj(edrop)
        if max_x >= 0:
            tmp = nn.Embedding(max_x + 1, hidden_channels)
            nn.init.orthogonal_(tmp.weight)
            self.xemb = nn.Sequential(tmp, nn.Dropout(dropout))
            in_channels = hidden_channels
        else:
            self.xemb = nn.Sequential(nn.Dropout(xdropout)) #nn.Identity()
            if not noinputlin and ("pure" in conv_fn or num_layers==0):
                self.xemb.append(nn.Linear(in_channels, hidden_channels))
                self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())
        self.res = res
        self.jk = jk
        if jk:
            self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers,))))
            
        if num_layers == 0 or conv_fn =="none":
            self.jk = False
            return
        convfn = convdict[conv_fn]
        if num_layers == 1:
            hidden_channels = out_channels
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()
        if "pure" in conv_fn:
            self.convs.append(convfn(hidden_channels, hidden_channels))
            for i in range(num_layers-1):
                self.lins.append(nn.Identity())
                self.convs.append(convfn(hidden_channels, hidden_channels))
            self.lins.append(nn.Dropout(taildropout, True))
        else:
            self.convs.append(convfn(in_channels, hidden_channels))
            self.lins.append(
                nn.Sequential(lnfn(hidden_channels, ln), nn.Dropout(dropout, True),
                            nn.ReLU(True)))
            for i in range(num_layers - 1):
                self.convs.append(
                    convfn(
                        hidden_channels,
                        hidden_channels if i == num_layers - 2 else out_channels))
                if i < num_layers - 2:
                    self.lins.append(
                        nn.Sequential(
                            lnfn(
                                hidden_channels if i == num_layers -
                                2 else out_channels, ln),
                            nn.Dropout(dropout, True), nn.ReLU(True)))
                else:
                    self.lins.append(nn.Identity())
        

    def forward(self, x, adj_t):
        x = self.xemb(x)
        jkx = []
        for i, conv in enumerate(self.convs):
            x1 = self.lins[i](conv(x, self.adjdrop(adj_t)))
            if self.res and x1.shape[-1] == x.shape[-1]:
                x = x1 + x
            else:
                x = x1
            if self.jk:
                jkx.append(x)
        if self.jk:
            jkx = torch.stack(jkx, dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            # print(sftmax)
            x = torch.sum(jkx*sftmax, dim=0)
        return x


# -------- works for graph property prediction --------
# modified from https://github.com/diningphil/gnn-comparison/blob/master/models/graph_classifiers/GIN.py
class GIN(torch.nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=4, dropout=0., train_eps=True,
                 pooling='mean', **kwargs):
        super(GIN, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.embeddings_dim = [hidden_channels] * num_layers
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []

        if pooling == 'sum':
            self.pooling = global_add_pool
        elif pooling == 'mean':
            self.pooling = global_mean_pool
        elif pooling == 'mmax':
            self.pooling = global_max_pool

        for layer, out_emb_dim in enumerate(self.embeddings_dim):

            if layer == 0:
                self.first_h = Sequential(Linear(in_channels, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                    Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU())
                self.linears.append(Linear(out_emb_dim, out_channels))
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.nns.append(Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU(),
                                      Linear(out_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()))
                self.convs.append(GINConv(self.nns[-1], train_eps=train_eps))

                self.linears.append(Linear(out_emb_dim, out_channels))

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0

        for layer in range(self.num_layers):
            if layer == 0:
                x = self.first_h(x)
                out += F.dropout(self.pooling(self.linears[layer](x), batch), p=self.dropout)
            else:
                # Layer l ("convolution" layer)
                x = self.convs[layer-1](x, edge_index)
                out += F.dropout(self.linears[layer](self.pooling(x, batch)), p=self.dropout, training=self.training)

        return out


# modified from https://github.com/diningphil/gnn-comparison/blob/master/models/graph_classifiers/GraphSAGE.py
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=4, dropout=0., 
                 aggregation='mean', pooling='mean', **kwargs):
        super().__init__()

        self.dropout = dropout
        self.aggregation = aggregation# can be mean | max | add

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(hidden_channels, hidden_channels)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = in_channels if i == 0 else hidden_channels

            # Overwrite aggregation method (default is set to mean
            conv = SAGEConv(dim_input, hidden_channels, aggr=self.aggregation)

            self.layers.append(conv)

        # For graph classification
        if pooling == 'sum':
            self.pooling = global_add_pool
        elif pooling == 'mean':
            self.pooling = global_mean_pool
        elif pooling == 'mmax':
            self.pooling = global_max_pool

        self.fc1 = nn.Linear(num_layers * hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_all = []
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == 'max':
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = self.pooling(x, batch)

        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout)
        x = self.fc2(x)
        return x


# -------- works for sub-graph based node classification --------
class GCNWithPooling(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128, dropout=0.5, pool='sum',
                 subset=True, subset_type='top', k=1, predictor_type=None, **kwargs):
        super().__init__()

        out_dim = out_channels if predictor_type is None else hidden_channels
        self.graph_encoder = SimpleGCN(in_channels, hidden_channels, out_dim, dropout)
        self.pooling = PoolingLayer(pool, subset, subset_type, k)

        ### Predictor layers
        self.predictor_type = predictor_type
        if predictor_type is not None:
            if predictor_type == 'ffn':
                self.predictor = torch.nn.Sequential(
                    torch.nn.Linear(2 * hidden_channels, hidden_channels),
                    torch.nn.ReLU(), 
                    torch.nn.Dropout(dropout), 
                    torch.nn.Linear(hidden_channels, out_channels)
                )
            else:
                raise NotImplementedError(f'Unknown predictor_type {predictor_type}')
        else:
            self.predictor = torch.nn.Identity()

    def forward(self, batched_data):
        h, h_hidden = self.graph_encoder(batched_data)
        if self.predictor_type is not None:
            h = torch.cat((h, h_hidden), dim=1)

        h = self.pooling(h, batched_data.batch)
        h = self.predictor(h)
        return h


# -------- works for graph property prediction on molecules --------
class GINVirtualnode(torch.nn.Module):

    def __init__(self, num_tasks, num_layer=5, emb_dim=300, residual=False, edge_feat=False,
                 drop_ratio=0.5, JK="last", graph_pooling="mean", **kwargs):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GINVirtualnode, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = GINNodeVirtualnode(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, 
                                           residual=residual, edge_feat=edge_feat)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)