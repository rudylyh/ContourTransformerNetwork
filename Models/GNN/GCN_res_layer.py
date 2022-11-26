import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from .GCN_layer  import GraphConvolution


class GraphResConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_dim, name=''):
        super(GraphResConvolution, self).__init__()
        self.in_dim = in_dim

        self.gcn_1 = GraphConvolution(in_dim, name='%s_1' % name)
        self.gcn_2 = GraphConvolution(in_dim, name='%s_2' % name)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.name = name

    def forward(self, input, adj):

        output_1 = self.gcn_1(input, adj)
        output_1_relu = self.relu1(output_1)

        output_2 = self.gcn_2(output_1_relu, adj)

        output_2_res = output_2 + input

        output = self.relu2(output_2_res)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.name + ')'