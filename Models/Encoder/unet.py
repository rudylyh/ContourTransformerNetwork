# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
import torch.nn.functional as F
import Models.GNN.utils as GNNUtils
import torchvision.transforms as transforms


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, nInputChannels=3, encoder_final_dim=32, gcn_in_dim=32, opts=None):
        super(UNet, self).__init__()
        self.encoder_final_dim = encoder_final_dim
        self.gcn_in_dim = gcn_in_dim
        self.opts = opts

        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

        self.inc = inconv(nInputChannels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.up1 = up(768, 256, bilinear=True)
        self.up2 = up(384, 128, bilinear=True)
        self.up3 = up(192, 64, bilinear=True)
        self.up4 = up(96, self.encoder_final_dim, bilinear=True)
        # self.outc = outconv(64, n_classes)

        edge_annotation_cnn_tunner_1 = nn.Conv2d(self.encoder_final_dim, self.gcn_in_dim, kernel_size=3, padding=1, bias=False)
        edge_annotation_cnn_tunner_bn_1 = nn.BatchNorm2d(self.gcn_in_dim)
        edge_annotation_cnn_tunner_relu_1 = nn.ReLU(inplace=True)
        edge_annotation_cnn_tunner_2 = nn.Conv2d(self.gcn_in_dim, self.gcn_in_dim, kernel_size=3, padding=1, bias=False)
        edge_annotation_cnn_tunner_bn_2 = nn.BatchNorm2d(self.gcn_in_dim)
        edge_annotation_cnn_tunner_relu_2 = nn.ReLU(inplace=True)
        self.edge_annotation_concat = nn.Sequential(edge_annotation_cnn_tunner_1,
                                                    edge_annotation_cnn_tunner_bn_1,
                                                    edge_annotation_cnn_tunner_relu_1,
                                                    edge_annotation_cnn_tunner_2,
                                                    edge_annotation_cnn_tunner_bn_2,
                                                    edge_annotation_cnn_tunner_relu_2
                                                    )

    def forward(self, x):
        x = self.normalize(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

    def pregcn_cnn(self, feature):
        feature = self.edge_annotation_concat(feature)
        if 'multicontour' in self.opts.keys() and self.opts['multicontour']:
            return feature.permute(0, 2, 3, 1).view(-1, self.opts['grid_size_matching'] ** 2, self.gcn_in_dim // 7)
        else:
            return feature.permute(0, 2, 3, 1).view(-1, self.opts['grid_size_matching'] ** 2, self.gcn_in_dim)

    def sampling(self, ids, features):
        cnn_out_feature = []

        for i in range(ids.size()[1]):
            id = ids[:, i, :]
            cnn_out = GNNUtils.gather_feature(id, features[i])
            cnn_out_feature.append(cnn_out)

        concat_features = torch.cat(cnn_out_feature, dim=2)

        return concat_features

    def reload(self, path):
        print('UNet. Ignore reloading.')

    def normalize(self, x):
        individual = torch.unbind(x, dim=0)
        out = []

        for x in individual:
            out.append(self.normalizer(x))

        return torch.stack(out, dim=0)


def unet(**kwargs):
    return UNet(**kwargs)
