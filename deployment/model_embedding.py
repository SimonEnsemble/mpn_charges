
# // ===============================
# // AUTHOR     : Ali Raza
# // CREATE DATE     : Dec 23, 2019
# // PURPOSE     : contains network that uses GNN
# // SPECIAL NOTES:
# // ===============================
# // Change History: 1.0: a simple network including GatedGraphConv
# // Change History: 1.3: Added: model for soft constraints
# // Change History: 1.6: Added: Handling minibatches.
# // Change History: 2.0: Added: mean correction and gaussian correction
# //
# //==================================
__author__ = "Ali Raza"
__copyright__ = "Copyright 2019"
__credits__ = []
__license__ = ""
__version__ = "1.0"
__maintainer__ = "ali raza"
__email__ = "razaa@oregonstate.edu"
__status__ = "done"

import torch
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform
import torch_scatter as ts
import copy



class Net_vanilla(torch.nn.Module):
    def __init__(self, NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE ):
        print("GNN_LAYERS = ", GNN_LAYERS)
        print("EMBEDDING_SIZE = ", EMBEDDING_SIZE)
        print("HIDDEN_FEATURES_SIZE = ", HIDDEN_FEATURES_SIZE)
        
        super(Net_vanilla, self).__init__()
        self.lin0 = torch.nn.Linear(NUM_NODE_FEATURES, EMBEDDING_SIZE,bias=False) # for embedding
        self.conv1 = GatedGraphConv(HIDDEN_FEATURES_SIZE, GNN_LAYERS)
        self.lin1 = torch.nn.Linear(HIDDEN_FEATURES_SIZE, 1)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = torch.sigmoid(self.lin0(x)) # embedding
        x = self.conv1(x1, edge_index)
        x = F.relu(x)
        x = self.lin1(x)

        return x.squeeze(1), x1.squeeze(1), None

class Net_mean_correction(torch.nn.Module):
    def __init__(self, NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE):
        print("GNN_LAYERS = ", GNN_LAYERS)
        print("EMBEDDING_SIZE = ", EMBEDDING_SIZE)
        print("HIDDEN_FEATURES_SIZE = ", HIDDEN_FEATURES_SIZE)
        
        super(Net_mean_correction, self).__init__()
        self.lin0 = torch.nn.Linear(NUM_NODE_FEATURES, EMBEDDING_SIZE,bias=False) # for embedding
        self.conv1 = GatedGraphConv(HIDDEN_FEATURES_SIZE, GNN_LAYERS)
        self.lin1 = torch.nn.Linear(HIDDEN_FEATURES_SIZE, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = torch.sigmoid(self.lin0(x)) # embedding
        x = self.conv1(x1, edge_index) 
        x = F.relu(x)
        x = self.lin1(x)
        mean_all = ts.scatter_mean(x, data.batch, dim=0)
        for i in range(0,data.num_graphs):
            x[data.batch==i] = x[data.batch==i]- mean_all[i]
        return x.squeeze(1), x1.squeeze(1), None, None


class Net_gaussian_correction(torch.nn.Module):
    def __init__(self, NUM_NODE_FEATURES,EMBEDDING_SIZE,GNN_LAYERS,HIDDEN_FEATURES_SIZE):
        print("GNN_LAYERS = ", GNN_LAYERS)
        print("EMBEDDING_SIZE = ", EMBEDDING_SIZE)
        print("HIDDEN_FEATURES_SIZE = ", HIDDEN_FEATURES_SIZE)
        
        super(Net_gaussian_correction, self).__init__()
        self.lin0 = torch.nn.Linear(NUM_NODE_FEATURES, EMBEDDING_SIZE,bias=False) # for embedding
        self.conv1 = GatedGraphConv(HIDDEN_FEATURES_SIZE, GNN_LAYERS)
        self.lin1 = torch.nn.Linear(HIDDEN_FEATURES_SIZE, 1)
        self.lin2 = torch.nn.Linear(HIDDEN_FEATURES_SIZE, 1)
        self.softplus = torch.nn.Softplus()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = torch.sigmoid(self.lin0(x)) # embedding
        x = self.conv1(x1, edge_index)
        x = F.relu(x)
        mu = self.lin1(x)
        sigma = self.softplus(self.lin2(x))
        uncorrected_mu = mu.clone()
        mu_all = ts.scatter_add(mu, data.batch, dim=0)
        sigma_all = ts.scatter_add(sigma, data.batch, dim=0)

        for i in range(0, data.num_graphs):
            mu[data.batch == i] = mu[data.batch == i] - mu_all[i] * (sigma[data.batch == i] / sigma_all[i])
        return mu.squeeze(1), x1.squeeze(1), sigma.squeeze(1), uncorrected_mu.squeeze(1)




# credit to https://github.com/rusty1s/pytorch_geometric/
class GatedGraphConv(MessagePassing):
    """The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`
    """

    def __init__(self,
                 out_channels,
                 num_layers,
                 aggr='add',
                 bias=True,
                 **kwargs):
        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()


    def forward(self, x, edge_index, edge_weight=None):
        """"""
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        if h.size(1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(h, self.weight[i])
            m = self.propagate(edge_index, x=m, edge_weight=edge_weight)
            h = self.rnn(m, h)

        return h


    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def __repr__(self):
        return '{}({}, num_layers={})'.format(
            self.__class__.__name__, self.out_channels, self.num_layers)


