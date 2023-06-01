import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, sort_edge_index, degree
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import HGTConv, Linear
import copy
from functools import partial


def mask(x, mask_rate=0.5):
    num_nodes = x.size(0)
    perm = torch.randperm(num_nodes, device=x.device)
    num_mask_nodes = int(mask_rate * num_nodes)

    mask_nodes = perm[: num_mask_nodes]

    return mask_nodes


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    # loss = -(x * y).sum(dim=-1)
    loss = loss.mean()
    return loss


class GAT(MessagePassing):
    def __init__(self, in_channels, dropout, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.dropout = dropout

        self.att_src = nn.Parameter(torch.Tensor(1, in_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x, edge_index):
        x_src, x_dst = x

        alpha_src = (x_src * self.att_src).sum(-1)
        alpha_dst = (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class Cross_View(nn.Module):
    def __init__(self, data, hidden_dim, feat_drop, alpha, att_drop1, att_drop2, r1, r2
                 , r3,
                 # r5, r6
                 ):
        super(Cross_View, self).__init__()
        self.data = data
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        # self.r4 = r4
        # self.r5 = r5
        # self.r6 = r6
        self.fc = nn.ModuleDict({
            n_type: nn.Linear(
                data[n_type].x.shape[1],
                hidden_dim,
                bias=True
            )
            for n_type in data.use_nodes
        })

        self.feat_drop = nn.Dropout(feat_drop)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        # mask gnn
        self.intra = nn.ModuleList([
            GAT(hidden_dim, att_drop1)
            for _ in range(len(self.data.schema_dict))
        ])
        self.action = nn.ModuleList([
            nn.PReLU() for _ in range(len(self.data.schema_dict))
        ])
        self.act = nn.ModuleDict({
            s: nn.PReLU() for s in self.data.use_nodes
        })

        self.bn = nn.ModuleList([
            BatchNorm(hidden_dim) for _ in range(len(self.data.schema_dict))
        ])
        self.schema_dict = {s: i for i, s in enumerate(self.data.schema_dict)}
        self.reset_parameter()
        # message passing gnn
        self.intra_mp = nn.ModuleList([
            GAT(hidden_dim, att_drop2)
            for _ in range(len(self.data.mp))
        ])
        self.action_mp = nn.ModuleList([
            nn.PReLU() for _ in range(len(self.data.mp))
        ])
        self.bn_mp = nn.ModuleList([
            BatchNorm(hidden_dim) for _ in range(len(self.data.mp))
        ])
        self.mp = {s: i for i, s in enumerate(self.data.mp)}
        # self.conv1 = GCNConv(data[data.main_node].x.size(1), 64)
        # self.act1 = nn.PReLU()
        # self.bn2 = BatchNorm(64)

    def reset_parameter(self):
        for fc in self.fc.values():
            nn.init.xavier_normal_(fc.weight, gain=1.414)

    def forward(self, data):

        h = {}
        for n_type in data.use_nodes:
            h[n_type] = self.act[n_type](
                self.feat_drop(
                    self.fc[n_type](data[n_type].x)
                )
            )
        # data[('paper', 'rev_writes', 'author')].edge_index = \
        #     data[('author', 'writes', 'paper')].edge_index[[1, 0]]
        # data[('paper', 'has_topic', 'field_of_study')].edge_index = \
        #     data[('field_of_study', 'rev_has_topic', 'paper')].edge_index[[1, 0]]
        # data[('paper', 'cite', 'R')].edge_index = data[('R', 'rev_cite', 'paper')].edge_index[[1, 0]]
        # Message Passing
        for n_type in data.mp:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra_mp[self.mp[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn_mp[self.mp[n_type]](embed1)
            h[dst] = self.action_mp[self.mp[n_type]](embed1)

        #########################################################################
        # Reconstruct
        # mask_node = mask(h[data.main_node], mask_rate=self.r1)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # # sc = ('a', 'p')
        # # sc = ('actor', 'movie')
        # sc = ('A', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # # sc = ('s', 'p')
        # # sc = ('director', 'movie')
        # sc = ('R', 'P')
        # src, dst = sc
        # x = h[src], h[dst]
        # # edge_index, edge_mask = dropout_edge(data[sc].edge_index, 0.1)
        # # embed2 = self.intra[self.schema_dict[sc]](x, edge_index)
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss1 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
        # ###########################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r2)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # # sc = ('s', 'p')
        # # sc = ('director', 'movie')
        # sc = ('R', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # # sc = ('a', 'p')
        # # sc = ('actor', 'movie')
        # sc = ('A', 'P')
        # src, dst = sc
        # x = h[src], h[dst]
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss2 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
        # # return self.alpha * loss1 + (1-self.alpha)*loss2
        # # #########################################################################
        # # # Aminer1
        # mask_node = mask(h[data.main_node], mask_rate=self.r3)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # sc = ('C', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # # sc = ('a', 'p')
        # # sc = ('actor', 'movie')
        # sc = ('A', 'P')
        # src, dst = sc
        # x = h[src], h[dst]
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss3 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
        # #####################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r4)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # sc = ('A', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # # sc = ('a', 'p')
        # # sc = ('actor', 'movie')
        # sc = ('C', 'P')
        # src, dst = sc
        # x = h[src], h[dst]
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss4 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
        # #########################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r5)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # sc = ('C', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # # sc = ('a', 'p')
        # # sc = ('actor', 'movie')
        # sc = ('R', 'P')
        # src, dst = sc
        # x = h[src], h[dst]
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss5 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
        # ###################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r6)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # sc = ('R', 'P')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # # sc = ('a', 'p')
        # # sc = ('actor', 'movie')
        # sc = ('C', 'P')
        # src, dst = sc
        # x = h[src], h[dst]
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss6 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
        # return loss1 + loss2 + loss4 + loss5 + loss6 + loss3
        # Aminer
        ########################################################################
        mask_node = mask(h[data.main_node], mask_rate=self.r1)
        main_h = h[data.main_node].clone()
        main_h[mask_node] = 0.0
        main_h[mask_node] += self.enc_mask_token
        h1 = 0
        for n_type in data.schema_dict1:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn[self.schema_dict[n_type]](embed1)
            h1 += self.action[self.schema_dict[n_type]](embed1)
        sc = ('C', 'P')
        src, dst = sc
        x = h[src], main_h
        embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        embed2 = self.bn[self.schema_dict[sc]](embed2)
        embed2 = self.action[self.schema_dict[sc]](embed2)
        loss1 = sce_loss(embed2[mask_node], h1[mask_node].detach())
        ##########################################################################
        mask_node = mask(h[data.main_node], mask_rate=self.r2)
        main_h = h[data.main_node].clone()
        main_h[mask_node] = 0.0
        main_h[mask_node] += self.enc_mask_token
        h1 = 0
        for n_type in data.schema_dict2:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn[self.schema_dict[n_type]](embed1)
            h1 += self.action[self.schema_dict[n_type]](embed1)
        sc = ('R', 'P')
        src, dst = sc
        x = h[src], main_h
        embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        embed2 = self.bn[self.schema_dict[sc]](embed2)
        embed2 = self.action[self.schema_dict[sc]](embed2)
        loss2 = sce_loss(embed2[mask_node], h1[mask_node].detach())
        ##########################################################################
        mask_node = mask(h[data.main_node], mask_rate=self.r3)
        main_h = h[data.main_node].clone()
        main_h[mask_node] = 0.0
        main_h[mask_node] += self.enc_mask_token
        h1 = 0
        for n_type in data.schema_dict3:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn[self.schema_dict[n_type]](embed1)
            h1 += self.action[self.schema_dict[n_type]](embed1)
        sc = ('A', 'P')
        src, dst = sc
        x = h[src], main_h
        embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        embed2 = self.bn[self.schema_dict[sc]](embed2)
        embed2 = self.action[self.schema_dict[sc]](embed2)
        loss3 = sce_loss(embed2[mask_node], h1[mask_node].detach())

        return loss1 + loss2 + loss3
        # MAG
        ############################################################################
        # data[('paper', 'rev_cites', 'paper')].edge_index = data[('paper', 'cites', 'paper')].edge_index[[1, 0]]
        # mask_node = mask(h[data.main_node], mask_rate=self.r1)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict1:
        #     src, _, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('R', 'rev_cite', 'paper')
        # src, _, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        # loss1 = sce_loss(embed1[mask_node], h1[mask_node].detach())
        # ###########################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r2)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict2:
        #     src, _, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('field_of_study', 'rev_has_topic', 'paper')
        # src, _, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # loss2 = sce_loss(embed1[mask_node], h1[mask_node].detach())
        # #####################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r3)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict3:
        #     src, _, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('author', 'writes', 'paper')
        # src, _, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # loss3 = sce_loss(embed1[mask_node], h1[mask_node].detach())
        #
        # return loss1 + loss2 + loss3
        #####################################################################
        # Freebase
        # mask_node = mask(h[data.main_node], mask_rate=self.r1)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict1:
        #     src, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('w', 'm')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        # loss1 = sce_loss(embed1[mask_node], h1[mask_node].detach())
        # ###########################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r2)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict2:
        #     src, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('d', 'm')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # loss2 = sce_loss(embed1[mask_node], h1[mask_node].detach())
        # #####################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r3)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict3:
        #     src, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('a', 'm')
        # src, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # loss3 = sce_loss(embed1[mask_node], h1[mask_node].detach())
        #
        # return loss1 + loss2 + loss3
        ############################################################################
        # CS
        # mask_node = mask(h[data.main_node], mask_rate=self.r1)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict1:
        #     src, _, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('paper', 'cite', 'paper')
        # src, _, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        # loss1 = sce_loss(embed1[mask_node], h1[mask_node].detach())
        # ###########################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r2)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict2:
        #     src, _, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('venue', 'Conference', 'paper')
        # src, _, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # loss2 = sce_loss(embed1[mask_node], h1[mask_node].detach())
        # #####################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r3)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict3:
        #     src, _, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('author', 'AP_write_last', 'paper')
        # src, _, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # loss3 = sce_loss(embed1[mask_node], h1[mask_node].detach())
        # #####################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r4)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        # h1 = 0
        # for n_type in data.schema_dict3:
        #     src, _, dst = n_type
        #     x = h[src], h[dst]
        #     embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
        #     embed1 = self.bn[self.schema_dict[n_type]](embed1)
        #     h1 += self.action[self.schema_dict[n_type]](embed1)
        # sc = ('field', 'in_L0', 'paper')
        # src, _, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # loss4 = sce_loss(embed1[mask_node], h1[mask_node].detach())
        #
        # return loss1 + loss2 + loss3 + loss4
        ##################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r1)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # sc = ('A', 'write', 'P')
        # src, _, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # sc = ('R', 'rev_cite', 'P')
        # src, _, dst = sc
        # x = h[src], h[dst]
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss1 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
        # ###########################################################################
        # mask_node = mask(h[data.main_node], mask_rate=self.r2)
        # main_h = h[data.main_node].clone()
        # main_h[mask_node] = 0.0
        # main_h[mask_node] += self.enc_mask_token
        #
        # sc = ('R', 'rev_cite', 'P')
        # src, _, dst = sc
        # x = h[src], main_h
        # embed1 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed1 = self.bn[self.schema_dict[sc]](embed1)
        # embed1 = self.action[self.schema_dict[sc]](embed1)
        #
        # sc = ('A', 'write', 'P')
        # src, _, dst = sc
        # x = h[src], h[dst]
        # embed2 = self.intra[self.schema_dict[sc]](x, data[sc].edge_index)
        # embed2 = self.bn[self.schema_dict[sc]](embed2)
        # embed2 = self.action[self.schema_dict[sc]](embed2)
        # loss2 = sce_loss(embed1[mask_node], embed2[mask_node].detach())
        # return self.alpha * loss1 + (1-self.alpha)*loss2

    def get_embed(self, data):
        h = {}
        for n_type in data.use_nodes:
            h[n_type] = self.act[n_type](
                self.feat_drop(
                    self.fc[n_type](data[n_type].x)
                )
            )
        # data[('paper', 'rev_writes', 'author')].edge_index = \
        #     data[('author', 'writes', 'paper')].edge_index[[1, 0]]
        # data[('paper', 'has_topic', 'field_of_study')].edge_index = \
        #     data[('field_of_study', 'rev_has_topic', 'paper')].edge_index[[1, 0]]
        # data[('paper', 'cite', 'R')].edge_index = \
        #     data[('R', 'rev_cite', 'paper')].edge_index[[1, 0]]

        h2 = 0.0
        for n_type in data.mp:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra_mp[self.mp[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn_mp[self.mp[n_type]](embed1)
            h[dst] = self.action_mp[self.mp[n_type]](embed1)

        for n_type in data.schema_dict:
            src, dst = n_type
            x = h[src], h[dst]
            embed1 = self.intra[self.schema_dict[n_type]](x, data[n_type].edge_index)
            embed1 = self.bn[self.schema_dict[n_type]](embed1)
            h2 += self.action[self.schema_dict[n_type]](embed1)
        return h2.detach()


class Encoder1(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, num_layers):
        super(Encoder1, self).__init__()
        self.num_layers = num_layers
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.act = nn.ModuleList()
        for layer in range(num_layers):  # excluding the input layer
            self.act.append(nn.PReLU())
            if layer == 0 and num_layers == 1:
                self.conv.append(GCNConv(in_dim, out_dim))
                self.bn.append(BatchNorm(out_dim))
            elif layer == 0:
                self.conv.append(GCNConv(in_dim, hidden))
                self.bn.append(BatchNorm(hidden))
            else:
                self.conv.append(GCNConv(hidden, out_dim))
                self.bn.append(BatchNorm(out_dim))

    def forward(self, h, edge_index):
        for i, layer in enumerate(self.conv):
            h = layer(h, edge_index)
            h = self.bn[i](h)
            h = self.act[i](h)

        return h

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.conv[i].reset_parameters()
            self.bn[i].reset_parameters()


class CG(nn.Module):
    def __init__(self, in_dim, out_dim, rate, hidden, layers):
        super(CG, self).__init__()
        self.online_encoder = Encoder1(in_dim, out_dim, hidden, layers)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.reset_parameters()
        self.rate = rate
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.criterion = self.setup_loss_fn("sce", 1)
        self.decoder = Encoder1(out_dim, in_dim, hidden, 1)

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, data):

        mask_nodes = mask(data[data.main_node].x, mask_rate=self.rate)
        x = data[data.main_node].x.clone()
        x[mask_nodes] = 0.0
        x[mask_nodes] += self.enc_mask_token

        h1 = self.online_encoder(x, data[('movie', 'metapath_1', 'movie')].edge_index)
        h1[mask_nodes] = 0.0
        re_x = self.decoder(h1, data[('movie', 'metapath_1', 'movie')].edge_index)
        loss = self.criterion(re_x[mask_nodes], data[data.main_node].x[mask_nodes].detach())

        return loss

    def get_embed(self, x, edge_index):
        h1 = self.online_encoder(x, edge_index)

        return h1.detach()
