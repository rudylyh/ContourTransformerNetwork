from .resnet import ResNet, Bottleneck
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import Models.GNN.utils as GNNUtils

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SkipResnet50(nn.Module):
    def __init__(self, concat_channels=64,
                 encoder_final_dim=512,
                 gcn_in_dim=256,
                 nInputChannels=3,
                 classifier="",
                 opts=None):

        super(SkipResnet50, self).__init__()

        # Default transform for all torchvision models
        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.device = torch.device("cuda")
        self.concat_channels = concat_channels
        self.nInputChannels = nInputChannels
        self.classifier = classifier
        self.resnet = ResNet(Bottleneck,
                             layers=[3, 4, 6, 3],
                             strides=[1, 2, 1, 1],
                             nInputChannels=nInputChannels,
                             dilations=[1, 1, 2, 4],
                             classifier=self.classifier)
        self.opts = opts
        self.encoder_final_dim = encoder_final_dim
        self.gcn_in_dim = gcn_in_dim

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
                                                    edge_annotation_cnn_tunner_relu_2)

    def reload(self, path):
        if self.nInputChannels != 3:
            print("Reloading resnet for: ", path, ", InputChannel: ", self.nInputChannels)
            model_full = ResNet(Bottleneck, layers=[3, 4, 6, 3], strides=[1, 2, 1, 1],
                             nInputChannels=3,
                             dilations=[1, 1, 2, 4]).to(self.device)
            model_full.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
            self.resnet.load_pretrained_ms(model_full, nInputChannels=self.nInputChannels)
            del(model_full)
        else:
            print("Reloading resnet from: ", path)
            self.resnet.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage), strict=False)

    def pregcn_cnn(self, feature):
        feature = self.edge_annotation_concat(feature)
        if not isinstance(self.opts['grid_size_encoder'], list):
            return feature.permute(0, 2, 3, 1).view(-1, self.opts['grid_size_encoder']**2, self.gcn_in_dim)
        elif len(self.opts['grid_size_encoder']) == 2:
            return feature.permute(0, 2, 3, 1).view(-1, self.opts['grid_size_encoder'][0]*self.opts['grid_size_encoder'][1], self.gcn_in_dim)

    def forward(self, x, init_control):
        # x = self.normalize(x)
        layer5_f = self.resnet(x, init_control)

        return layer5_f

    def sampling(self, ids, features):
        cnn_out_feature = []

        for i in range(ids.size()[1]):
            id = ids[:, i, :]
            cnn_out = GNNUtils.gather_feature(id, features[i])
            cnn_out_feature.append(cnn_out)

        concat_features = torch.cat(cnn_out_feature, dim=2)

        return concat_features

    def sampling_multicontour(self, ids, ids_left, ids_right, features):
        cnn_out_feature = []

        for i in reversed(range(ids_left.size()[1])):
            id = ids_left[:, i, :]
            cnn_out = GNNUtils.gather_feature(id, features[0])
            cnn_out_feature.append(cnn_out)

        for i in range(ids.size()[1]):
            id = ids[:, i, :]
            cnn_out = GNNUtils.gather_feature(id, features[0])
            cnn_out_feature.append(cnn_out)

        for i in range(ids_right.size()[1]):
            id = ids_right[:, i, :]
            cnn_out = GNNUtils.gather_feature(id, features[0])
            cnn_out_feature.append(cnn_out)

        concat_features = torch.cat(cnn_out_feature, dim=2)

        return concat_features

    def normalize(self, x):
        individual = torch.unbind(x, dim=0)
        out = []

        for x in individual:
            out.append(self.normalizer(x))

        return torch.stack(out, dim=0)


if __name__ == '__main__':
    model = SkipResnet50()
    model(torch.randn(1, 3, 224, 224))
