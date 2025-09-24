from typing import Iterable, Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SortAggregation, GCNConv, SAGEConv, GATConv

import torch_sparse
from torch_sparse import SparseTensor
from torch_sparse.matmul import spmm_add

from models.utils import adjoverlap


# Edge dropout
class DropEdge(nn.Module):

    def __init__(self, dp: float = 0.0) -> None:
        super().__init__()
        self.dp = dp

    def forward(self, edge_index: Tensor):
        if self.dp == 0:
            return edge_index
        mask = torch.rand_like(edge_index[0], dtype=torch.float) > self.dp
        return edge_index[:, mask]


# Edge dropout with adjacency matrix as input
class DropAdj(nn.Module):
    doscale: Final[bool] # whether to rescale edge weight
    def __init__(self, dp: float = 0.0, doscale=True) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1/(1-dp)))
        self.doscale = doscale

    def forward(self, adj: SparseTensor)->SparseTensor:
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = torch_sparse.masked_select_nnz(adj, mask, layout="coo")
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)
        return adj


##################################################################
#################### simple MLP link predictor ###################
##################################################################

class MLPLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, **kwargs):
        super(MLPLinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1: 
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


##################################################################
#################### CFLP link predictor ###################
##################################################################

class CFLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, dec='hadamard', **kwargs):
        super(CFLinkPredictor, self).__init__()
        self.dec = dec
        if dec == 'innerproduct':
            dim_in = 2
        elif dec == 'hadamard':
            dim_in = in_channels + 1
        elif dec == 'mlp':
            dim_in = 1 + 2*in_channels

        self.mlp_out = nn.Sequential(
            nn.Linear(dim_in, hidden_channels, bias=True),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(hidden_channels, 1, bias=False)
        )

    def forward(self, z_i, z_j, T):
        if self.dec == 'innerproduct':
            z = (z_i * z_j).sum(1).view(-1, 1)
            h = torch.cat((z, T.view(-1, 1)), dim=1)
        elif self.dec == 'mlp':
            h = torch.cat((z_i, z_j, T.view(-1, 1)), dim=1)
        elif self.dec == 'hadamard':
            z = z_i * z_j
            h = torch.cat((z, T.view(-1, 1)), dim=1)

        h = self.mlp_out(h).squeeze()
        return h

    def reset_parameters(self):
        for lin in self.mlp_out:
            try:
                lin.reset_parameters()
            except:
                continue


##################################################################
#################### link predictors from NCN ####################
##################################################################

# GAE predictor
class LinkPredictor(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 **kwargs):
        super(LinkPredictor, self).__init__()

        self.lins = nn.Sequential()

        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        if num_layers == 1:
            self.lins = nn.Linear(in_channels, out_channels)
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.lins.append(lnfn(hidden_channels, ln))
            self.lins.append(nn.Dropout(dropout, inplace=True))
            self.lins.append(nn.ReLU(inplace=True))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.lins.append(lnfn(hidden_channels, ln))
                self.lins.append(nn.Dropout(dropout, inplace=True))
                self.lins.append(nn.ReLU(inplace=True))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = [0.25]):
        x = x[tar_ei].prod(dim=0)
        x = self.lins(x)
        return x.expand(-1, len(cndropprobs) + 1)

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])


# GAE + CN link predictor
class SCNLinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        xcn = cn.sum(dim=-1).float().reshape(-1, 1)
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij)],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])


# another GAE + CN predictor
class CatSCNLinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels+1, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        xcn = cn.sum(dim=-1).float().reshape(-1, 1)
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(torch.cat((xcn, xij), dim=-1) )],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])


# GAE + CN predictor boosted by CNC trick
class IncompleteSCN1Predictor(SCNLinkPredictor):
    learnablept: Final[bool]
    depth: Final[int]
    splitsize: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0,
                 alpha=1.0,
                 scale=5,
                 offset=3,
                 trainresdeg=8,
                 testresdeg=128,
                 pt=0.5,
                 learnablept=False,
                 depth=1,
                 splitsize=-1,
                 ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, edrop, ln, cndeg, use_xlin, tailact, twolayerlin, beta)
        self.learnablept= learnablept
        self.depth = depth
        self.splitsize = splitsize
        self.lins = nn.Sequential()
        self.register_buffer("alpha", torch.tensor([alpha]))
        self.register_buffer("pt", torch.tensor([pt]))
        self.register_buffer("scale", torch.tensor([scale]))
        self.register_buffer("offset", torch.tensor([offset]))

        self.trainresdeg = trainresdeg
        self.testresdeg = testresdeg
        self.ptlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, 1), nn.Sigmoid())
        # print(self.xcnlin)

    def clampprob(self, prob, pt):
        p0 = torch.sigmoid_(self.scale*(prob-self.offset))
        return self.alpha*pt*p0/(pt*p0+1-p0)

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = [],
                           depth: int=None):
        assert len(cndropprobs) == 0
        if depth is None:
            depth = self.depth
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        xij = xi*xj
        if depth > 0.5:
            cn, cnres1, cnres2 = adjoverlap(
                    adj,
                    adj,
                    tar_ei,
                    filled1,
                    calresadj=True,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)
        else:
            cn = adjoverlap(
                    adj,
                    adj,
                    tar_ei,
                    filled1,
                    calresadj=False,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)
        xcns = [cn.sum(dim=-1).float().reshape(-1, 1)]
        
        if depth > 0.5:
            potcn1 = cnres1.coo()
            potcn2 = cnres2.coo()
            with torch.no_grad():
                if self.splitsize < 0:
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = self.forward(
                        x, adj, ei1,
                        filled1, depth-1).flatten()
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = self.forward(
                        x, adj, ei2,
                        filled1, depth-1).flatten()
                else:
                    num1 = potcn1[1].shape[0]
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = torch.empty_like(potcn1[1], dtype=torch.float)
                    for i in range(0, num1, self.splitsize):
                        probcn1[i:i+self.splitsize] = self.forward(x, adj, ei1[:, i: i+self.splitsize], filled1, depth-1).flatten()
                    num2 = potcn2[1].shape[0]
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = torch.empty_like(potcn2[1], dtype=torch.float)
                    for i in range(0, num2, self.splitsize):
                        probcn2[i:i+self.splitsize] = self.forward(x, adj, ei2[:, i: i+self.splitsize],filled1, depth-1).flatten()
            if self.learnablept:
                pt = self.ptlin(xij)
                probcn1 = self.clampprob(probcn1, pt[potcn1[0]]) 
                probcn2 = self.clampprob(probcn2, pt[potcn2[0]])
            else:
                probcn1 = self.clampprob(probcn1, self.pt)
                probcn2 = self.clampprob(probcn2, self.pt)
            probcn1 = probcn1 * potcn1[-1]
            probcn2 = probcn2 * potcn2[-1]
            cnres1.set_value_(probcn1, layout="coo")
            cnres2.set_value_(probcn2, layout="coo")
            xcn1 = cnres1.sum(dim=-1).float().reshape(-1, 1)
            xcn2 = cnres2.sum(dim=-1).float().reshape(-1, 1)
            xcns[0] = xcns[0] + xcn2 + xcn1
        xij = self.xijlin(xij)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs

    def setalpha(self, alpha: float):
        self.alpha.fill_(alpha)
        print(f"set alpha: {alpha}")

    def forward(self,
                x,
                adj,
                tar_ei,
                filled1: bool = False,
                depth: int = None):
        if depth is None:
            depth = self.depth
        return self.multidomainforward(x, adj, tar_ei, filled1, [],
                                       depth)


# NCN predictor
class CNLinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0,
                 **kwargs):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        xcns = [spmm_add(cn, x)]
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])


# GAE predictor for ablation study
class CN0LinkPredictor(nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        self.cndeg = cndeg

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(xij)],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])


# NCNC predictor
class IncompleteCN1Predictor(CNLinkPredictor):
    learnablept: Final[bool]
    depth: Final[int]
    splitsize: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0,
                 alpha=1.0,
                 scale=5,
                 offset=3,
                 trainresdeg=8,
                 testresdeg=128,
                 pt=0.5,
                 learnablept=False,
                 depth=1,
                 splitsize=-1,
                 **kwargs
                 ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, edrop, ln, cndeg, use_xlin, tailact, twolayerlin, beta)
        self.learnablept= learnablept
        self.depth = depth
        self.splitsize = splitsize
        self.lins = nn.Sequential()
        self.register_buffer("alpha", torch.tensor([alpha]))
        self.register_buffer("pt", torch.tensor([pt]))
        self.register_buffer("scale", torch.tensor([scale]))
        self.register_buffer("offset", torch.tensor([offset]))

        self.trainresdeg = trainresdeg
        self.testresdeg = testresdeg
        self.ptlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, 1), nn.Sigmoid())

    def clampprob(self, prob, pt):
        p0 = torch.sigmoid_(self.scale*(prob-self.offset))
        return self.alpha*pt*p0/(pt*p0+1-p0)

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = [],
                           depth: int=None):
        assert len(cndropprobs) == 0
        if depth is None:
            depth = self.depth
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        xij = xi*xj
        x = x + self.xlin(x)
        if depth > 0.5:
            cn, cnres1, cnres2 = adjoverlap(
                    adj,
                    adj,
                    tar_ei,
                    filled1,
                    calresadj=True,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)
        else:
            cn = adjoverlap(
                    adj,
                    adj,
                    tar_ei,
                    filled1,
                    calresadj=False,
                    cnsampledeg=self.cndeg,
                    ressampledeg=self.trainresdeg if self.training else self.testresdeg)
        xcns = [spmm_add(cn, x)]
        
        if depth > 0.5:
            potcn1 = cnres1.coo()
            potcn2 = cnres2.coo()
            with torch.no_grad():
                if self.splitsize < 0:
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = self.forward(
                        x, adj, ei1,
                        filled1, depth-1).flatten()
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = self.forward(
                        x, adj, ei2,
                        filled1, depth-1).flatten()
                else:
                    num1 = potcn1[1].shape[0]
                    ei1 = torch.stack((tar_ei[1][potcn1[0]], potcn1[1]))
                    probcn1 = torch.empty_like(potcn1[1], dtype=torch.float)
                    for i in range(0, num1, self.splitsize):
                        probcn1[i:i+self.splitsize] = self.forward(x, adj, ei1[:, i: i+self.splitsize], filled1, depth-1).flatten()
                    num2 = potcn2[1].shape[0]
                    ei2 = torch.stack((tar_ei[0][potcn2[0]], potcn2[1]))
                    probcn2 = torch.empty_like(potcn2[1], dtype=torch.float)
                    for i in range(0, num2, self.splitsize):
                        probcn2[i:i+self.splitsize] = self.forward(x, adj, ei2[:, i: i+self.splitsize],filled1, depth-1).flatten()
            if self.learnablept:
                pt = self.ptlin(xij)
                probcn1 = self.clampprob(probcn1, pt[potcn1[0]]) 
                probcn2 = self.clampprob(probcn2, pt[potcn2[0]])
            else:
                probcn1 = self.clampprob(probcn1, self.pt)
                probcn2 = self.clampprob(probcn2, self.pt)
            probcn1 = probcn1 * potcn1[-1]
            probcn2 = probcn2 * potcn2[-1]
            cnres1.set_value_(probcn1, layout="coo")
            cnres2.set_value_(probcn2, layout="coo")
            xcn1 = spmm_add(cnres1, x)
            xcn2 = spmm_add(cnres2, x)
            xcns[0] = xcns[0] + xcn2 + xcn1
        
        xij = self.xijlin(xij)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs

    def setalpha(self, alpha: float):
        self.alpha.fill_(alpha)
        print(f"set alpha: {alpha}")

    def forward(self,
                x,
                adj,
                tar_ei,
                filled1: bool = False,
                depth: int = None):
        if depth is None:
            depth = self.depth
        return self.multidomainforward(x, adj, tar_ei, filled1, [],
                                       depth)


# NCN2 predictor
class CNhalf2LinkPredictor(CNLinkPredictor):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 ln=False,
                 tailact=False,
                 **kwargs):
        super().__init__(in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout, ln=ln, tailact=tailact, **kwargs)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()
        self.xcn12lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
            
    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        adj2 = adj@adj
        cn12 = adjoverlap(adj, adj2, tar_ei, filled1, cnsampledeg=self.cndeg)
        cn21 = adjoverlap(adj2, adj, tar_ei, filled1, cnsampledeg=self.cndeg)

        xcns = [(spmm_add(cn, x), spmm_add(cn12, x)+spmm_add(cn21, x))]
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn[0]) * self.beta + self.xcn12lin(xcn[1]) + xij) for xcn in xcns],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])


# NCN-diff
class CNResLinkPredictor(CNLinkPredictor):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 ln=False,
                 tailact=False,
                 **kwargs):
        super().__init__(in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout, ln=ln, tailact=tailact, **kwargs)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()
        self.xcnreslin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
            
    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        x = x + self.xlin(x)
        cn, cnres1, cnres2 = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg, calresadj=True)

        xcns = [(spmm_add(cn, x), spmm_add(cnres1, x)+spmm_add(cnres2, x))]
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn[0]) * self.beta + self.xcnreslin(xcn[1]) + xij) for xcn in xcns],
            dim=-1)
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])


# NCN with higher order neighborhood overlaps than NCN-2
class CN2LinkPredictor(nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1):
        super().__init__()

        self.lins = nn.Sequential()

        self.register_parameter("alpha", nn.Parameter(torch.ones((3))))
        self.register_parameter("beta", nn.Parameter(torch.ones((1))))
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xcn1lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn2lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xcn4lin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, out_channels))

    def forward(self, x, adj: SparseTensor, tar_ei, filled1: bool = False):
        spadj = adj.to_torch_sparse_coo_tensor()
        adj2 = SparseTensor.from_torch_sparse_coo_tensor(spadj @ spadj, False)
        cn1 = adjoverlap(adj, adj, tar_ei, filled1)
        cn2 = adjoverlap(adj, adj2, tar_ei, filled1)
        cn3 = adjoverlap(adj2, adj, tar_ei, filled1)
        cn4 = adjoverlap(adj2, adj2, tar_ei, filled1)
        xij = self.xijlin(x[tar_ei[0]] * x[tar_ei[1]])
        xcn1 = self.xcn1lin(spmm_add(cn1, x))
        xcn2 = self.xcn2lin(spmm_add(cn2, x))
        xcn3 = self.xcn2lin(spmm_add(cn3, x))
        xcn4 = self.xcn4lin(spmm_add(cn4, x))
        alpha = torch.sigmoid(self.alpha).cumprod(-1)
        x = self.lin(alpha[0] * xcn1 + alpha[1] * xcn2 * xcn3 +
                     alpha[2] * xcn4 + self.beta * xij)
        return x

##################################################################
#################### link predictors from SEAL ###################
##################################################################

class CNNLinkPredictor(nn.Module):
    def __init__(self, k=30, conv1d_channels=[16, 32], conv1d_kws=[64, 5]):
        super().__init__()
        self.pool = SortAggregation(k)
        self.conv1 = torch.nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = torch.nn.MaxPool1d(2, 2)
        self.conv2 = torch.nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = torch.nn.Linear(dense_dim, 128)
        self.lin2 = torch.nn.Linear(128, 1)
    
    def forward(self, x, batch):
        # Global pooling.
        x = self.pool(x, batch)  # global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x