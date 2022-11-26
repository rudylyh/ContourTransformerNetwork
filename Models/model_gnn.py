import sys
import torch
import torch.nn as nn
from Models.Encoder.resnet_GCN_skip import SkipResnet50
from Models.Encoder.unet import UNet
from Models.Encoder.hrnet import HighResolutionNet
from Models.Encoder.stackedhourglass import StackedHourGlass
from Models.GNN.GCN import GCN
from Models.GNN.Heatmap import Heatmap
import Models.GNN.utils as GNNUtils


class PolyGNN(nn.Module):
    def __init__(self,
                 gcn_out_dim=256,
                 gcn_steps=0,
                 get_point_annotation=False,
                 nInputChannels=3,
                 opts=None
                 ):
        super(PolyGNN, self).__init__()

        self.gcn_out_dim = gcn_out_dim
        self.gcn_steps = gcn_steps
        self.get_point_annotation = get_point_annotation
        self.device = torch.device("cuda")
        self.opts = opts

        print('Building GNN Encoder')

        if get_point_annotation:
            nInputChannels = 4
        else:
            nInputChannels = nInputChannels

        if self.opts['encoder'] == 'resnet':
            self.encoder = SkipResnet50(nInputChannels=nInputChannels,
                                        gcn_in_dim=self.opts['gcn_in_dim'],
                                        classifier=self.opts["psp"],
                                        opts=self.opts)
        elif self.opts['encoder'] == 'unet':
            self.encoder = UNet(nInputChannels=nInputChannels, gcn_in_dim=self.opts['gcn_in_dim'], opts=self.opts)
        elif self.opts['encoder'] == 'hourglass':
            self.encoder = StackedHourGlass(nJoints=self.opts['cp_num'], opts=self.opts)
        elif self.opts['encoder'] == 'hrnet':
            self.encoder = HighResolutionNet(self.opts)

        for k in range(self.opts['num_heads']):
            # Build GNN heads
            for step in range(self.gcn_steps):
                if step == 0 and k == 0:
                    self.gnns = nn.ModuleList(
                        [GCN(in_dim=self.encoder.gcn_in_dim,
                             out_dim=self.gcn_out_dim).to(self.device)])
                else:
                    self.gnns.append(GCN(in_dim=self.encoder.gcn_in_dim,
                                         out_dim=self.gcn_out_dim).to(self.device))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                nn.init.constant_(m.bias, 0)

    def reload(self, path, strict=False):
        print("Reloading full model from: ", path)
        self.load_state_dict(torch.load(path)['state_dict'], strict=strict)

    def forward(self, x, init_control, gcn_component):
        out_dict = {}
        out_dict['pred_coor'] = []
        out_dict['roi_pooled'] = []
        out_dict['init_control'] = []
        encoder_out = self.encoder.forward(x, init_control)
        # encoder_out = self.encoder.forward(x)

        if isinstance(encoder_out, tuple):
            heatmap_preds = encoder_out[0]
            encoder_out = encoder_out[1]

        flatted_out = self.encoder.pregcn_cnn(encoder_out)

        # For each head
        for i, (init_control_i, gcn_component_i) in enumerate(zip(init_control, gcn_component)):
            pred_head = []

            for s in range(self.gcn_steps):
                if s == 0:
                    adjacent = gcn_component_i['adj_matrix']
                    out_dict['init_control'].append(init_control_i)
                else:
                    init_control_i = pred_out

                parrallels = [init_control_i]
                cnn_feature = GNNUtils.interpolated_sum_multicontour(flatted_out, parrallels, self.opts['grid_size_encoder'])
                # return cnn_feature
                input_feature_i = cnn_feature
                pred = self.gnns[i*self.opts['num_heads'] + s].forward(input_feature_i, adjacent)

                if self.opts['normstep']:
                    # Normalize the steps
                    pred_norm = torch.norm(pred, dim=2).unsqueeze(-1)
                    pred_out = init_control_i + pred / pred_norm
                    pred_out = torch.clamp(pred_out, 0, 1)
                else:
                    pred_out = init_control_i + pred
                    pred_out = torch.clamp(pred_out, 0, 1)

                pred_head.append(pred_out)
            out_dict['pred_coor'].append(pred_head)

        return out_dict

