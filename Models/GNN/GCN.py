import torch
import torch.nn as nn
from .GCN_layer import GraphConvolution
from .GCN_res_layer import GraphResConvolution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(nn.Module):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 out_features=2):

        super(GCN, self).__init__()
        self.in_dim = in_dim

        self.gcn_0 = GraphConvolution(in_dim=self.in_dim, name='gcn_0', out_dim=out_dim)
        self.gcn_res_1 = GraphResConvolution(in_dim=out_dim, name='gcn_res_1')
        self.gcn_res_2 = GraphResConvolution(in_dim=out_dim, name='gcn_res_2')
        self.gcn_res_3 = GraphResConvolution(in_dim=out_dim, name='gcn_res_3')
        self.gcn_res_4 = GraphResConvolution(in_dim=out_dim, name='gcn_res_4')
        self.gcn_res_5 = GraphResConvolution(in_dim=out_dim, name='gcn_res_5')
        self.gcn_res_6 = GraphResConvolution(in_dim=out_dim, name='gcn_res_6')
        self.gcn_7 = GraphConvolution(in_dim=out_dim, name='gcn_7', out_dim=32)

        self.fc = nn.Linear(
            in_features=32,
            out_features=out_features,
        )

    def forward(self, input, adj):
        input = self.gcn_0(input, adj)
        input = self.gcn_res_1(input, adj)
        input = self.gcn_res_2(input, adj)
        input = self.gcn_res_3(input, adj)
        input = self.gcn_res_4(input, adj)
        input = self.gcn_res_5(input, adj)
        input = self.gcn_res_6(input, adj)
        output = self.gcn_7(input, adj)
        return self.fc(output)
