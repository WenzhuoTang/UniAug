import torch
from torch_geometric.nn import MessagePassing


# -------- GDSS utils --------
def mask_adjs(adjs, flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param flags: B x N
    :return:
    """
    if flags is None:
        flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs

def pow_tensor(x, cnum):
    # x : B x N x N
    x_ = x.clone()
    xc = [x.unsqueeze(1)]
    for _ in range(cnum-1):
        x_ = torch.bmm(x_, x)
        xc.append(x_.unsqueeze(1))
    xc = torch.cat(xc, dim=1)
    return xc


# -------- STA utils --------
class MessagePropRandomWalk(MessagePassing):
    def __init__(self, node_dim=-3):
        super().__init__(aggr='add', node_dim=node_dim)  

    def forward(self, x, edge_index, norm):
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm * x_j


class KeyPropRandomWalk(MessagePassing):
    def __init__(self, node_dim=-2):
        super().__init__(aggr='add', node_dim=node_dim)  

    def forward(self, x, edge_index, norm):
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm * x_j