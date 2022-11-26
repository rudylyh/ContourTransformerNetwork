import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_dim, out_dim=None, name='', ):
        super(GraphConvolution, self).__init__()
        self.in_dim = in_dim

        if out_dim == None:
            self.out_dim = in_dim
        else:
            self.out_dim = out_dim

        self.fc1 = nn.Linear(
            in_features=self.in_dim,
            out_features=self.out_dim,
        )

        self.fc2 = nn.Linear(
            in_features=self.in_dim,
            out_features=self.out_dim,
        )

        # self.fc1_0 = nn.Linear(
        #     in_features=self.in_dim,
        #     out_features=self.out_dim,
        # )
        # self.fc1_1 = nn.Linear(
        #     in_features=self.in_dim,
        #     out_features=self.out_dim,
        # )
        # self.fc1_2 = nn.Linear(
        #     in_features=self.in_dim,
        #     out_features=self.out_dim,
        # )
        # self.fc1_3 = nn.Linear(
        #     in_features=self.in_dim,
        #     out_features=self.out_dim,
        # )
        # self.fc1_4 = nn.Linear(
        #     in_features=self.in_dim,
        #     out_features=self.out_dim,
        # )
        # self.fc1_5 = nn.Linear(
        #     in_features=self.in_dim,
        #     out_features=self.out_dim,
        # )
        #
        # self.fc2_0 = nn.Linear(
        #     in_features=self.in_dim,
        #     out_features=self.out_dim,
        # )
        # self.fc2_1 = nn.Linear(
        #     in_features=self.in_dim,
        #     out_features=self.out_dim,
        # )
        # self.fc2_2 = nn.Linear(
        #     in_features=self.in_dim,
        #     out_features=self.out_dim,
        # )
        # self.fc2_3 = nn.Linear(
        #     in_features=self.in_dim,
        #     out_features=self.out_dim,
        # )
        # self.fc2_4 = nn.Linear(
        #     in_features=self.in_dim,
        #     out_features=self.out_dim,
        # )
        # self.fc2_5 = nn.Linear(
        #     in_features=self.in_dim,
        #     out_features=self.out_dim,
        # )

        self.name = name

    def forward(self, input, adj):
        # t0 = [0, 4, 9, 14, 19]
        # t1 = [5, 10, 15, 20]
        # t2 = [1, 6, 11, 16, 21]
        # t3 = [2, 7, 12, 17, 22]
        # t4 = [3, 8, 13, 18, 23]
        # t5 = [24, 25, 26, 27, 28, 29]
        # group_0 = input[:, t0, :]
        # group_1 = input[:, t1, :]
        # group_2 = input[:, t2, :]
        # group_3 = input[:, t3, :]
        # group_4 = input[:, t4, :]
        # group_5 = input[:, t5, :]
        #
        # state_in_0 = self.fc1_0(group_0)
        # state_in_1 = self.fc1_1(group_1)
        # state_in_2 = self.fc1_2(group_2)
        # state_in_3 = self.fc1_3(group_3)
        # state_in_4 = self.fc1_4(group_4)
        # state_in_5 = self.fc1_5(group_5)
        #
        # mul = torch.bmm(adj, input)
        # group_0 = mul[:, t0, :]
        # group_1 = mul[:, t1, :]
        # group_2 = mul[:, t2, :]
        # group_3 = mul[:, t3, :]
        # group_4 = mul[:, t4, :]
        # group_5 = mul[:, t5, :]
        #
        # forward_input_0 = self.fc2_0(group_0)
        # forward_input_1 = self.fc2_1(group_1)
        # forward_input_2 = self.fc2_2(group_2)
        # forward_input_3 = self.fc2_3(group_3)
        # forward_input_4 = self.fc2_4(group_4)
        # forward_input_5 = self.fc2_5(group_5)
        #
        # # forward_input = self.fc2(torch.bmm(adj, input))
        # state_in = torch.cat((state_in_0, state_in_1, state_in_2, state_in_3,
        #                       state_in_4, state_in_5), dim=1)
        # forward_input = torch.cat((forward_input_0, forward_input_1, forward_input_2, forward_input_3,
        #                            forward_input_4, forward_input_5), dim=1)
        state_in = self.fc1(input)
        forward_input = self.fc2(torch.bmm(adj, input))

        return state_in + forward_input

    def __repr__(self):
        return self.__class__.__name__ + ' (' +  self.name + ')'
