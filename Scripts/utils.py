import torch.optim as optim
import sys
import os
import torch
sys.path.append('.')
from DataProvider import knuckle_contour_data, lung_contour_data, knee_contour_data, hip_contour_data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg.basic import inv


def get_data_loaders(opts, mode, shuffle=True):
    '''
    Get data loaders based on selected dataset
    :param opts: opts loaded from configuration json file
    :return: train loader and validation loader
    '''

    print('Building dataloaders')
    if opts['data'] == 'knuckle':
        dataset = knuckle_contour_data
    elif opts['data'] == 'lung':
        dataset = lung_contour_data
    elif opts['data'] == 'knee':
        dataset = knee_contour_data
    elif opts['data'] == 'hip':
        dataset = hip_contour_data

    # if mode == 'train':
    #     shuffle = True
    # else:
    #     shuffle = False

    data = dataset.SelfDataset(mode=mode, opts=opts)
    loader = DataLoader(data, batch_size=opts['dataset'][mode]['batch_size'],
                        shuffle=shuffle,
                        num_workers=opts['dataset'][mode]['num_workers'],
                        pin_memory=True)
    return loader


def init_optimizer(optm, lr, weight_decay, model):
    '''
    Initialize optimizer
    :param optm: the selected optimizer
    :param lr: initial learning rate
    :param weight_decay: configured weight decay
    :param model: model to extract parameters that require gradient
    :return: optimizer
    '''

    no_wd = []
    wd = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            # No optimization for frozen params
            continue

        if 'bn' in name or 'bias' in name:
            no_wd.append(p)
        else:
            wd.append(p)

    if optm == 'adam':
        optimizer = optim.Adam(
            [
                {'params': no_wd, 'weight_decay': 0.0},
                {'params': wd}
            ],
            lr=lr,
            weight_decay=weight_decay,
            amsgrad=False)
    elif optm == 'sgd':
        optimizer = optim.SGD(
            [
                {'params': no_wd, 'weight_decay': 0.0},
                {'params': wd}
            ],
            lr=lr,
            weight_decay=weight_decay)
    return optimizer


# Find the contor that
def threshold_contour(ctr):
    '''
    Threshold the contour points based on the constructed spline points to make them equal height
    :param ctr: contour points
    :return: thresholded contour points
    '''

    middle_x = ctr[len(ctr) // 2, 0]
    middle_y = ctr[len(ctr) // 2, 1]
    left_points = ctr[ctr[:, 0] < middle_x, :]
    right_points = ctr[ctr[:, 0] > middle_x, :]
    if len(left_points) == 0 or len(right_points) == 0:
        return ctr
    left_max_row = max(left_points[:, 1])
    right_max_row = max(right_points[:, 1])
    if left_max_row <= right_max_row:
        ctr = ctr[ctr[:, 1] <= left_max_row]
    else:
        ctr = ctr[ctr[:, 1] <= right_max_row]
    return ctr


def create_folder(path):

    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s'%(path))
        print('Experiment folder created at: %s'%(path))




# Compute the TPS transformation matrix
def get_tps_l(sample_ctr):
    n = sample_ctr.shape[ 0 ]
    pt_dist = cdist( sample_ctr, sample_ctr, metric = "euclidean" )
    pt_dist[pt_dist == 0] = 1
    K = ( pt_dist ** 2 ) * np.log( pt_dist ** 2 )
    np.fill_diagonal(K, 0)
    P = np.hstack( ( np.ones( ( n, 1 ) ), sample_ctr ) )
    L = np.vstack( ( np.hstack( ( K, P ) ), np.hstack( ( P.T, np.zeros( ( 3, 3 ) ) ) ) ) )
    inv_L = np.linalg.inv(L)
    # L1 = L.astype(np.float32)
    # inv_L1 = inv_L.astype(np.float32)
    return L, inv_L


def cmp_img_vgg_feat(vgg, imgs, sample_imgs):
    best_indices = list()
    for n in range(0, imgs.shape[0]):
        best_idx = None
        min_dist = np.iinfo(np.int).max
        for i in range(0, len(sample_imgs)):
            img_feat = vgg(imgs[n].unsqueeze(0))
            sample_feat = vgg(sample_imgs[i].unsqueeze(0))
            dist = 0.0
            for j in range(len(img_feat)):
                dist += torch.dist(img_feat[j], sample_feat[j])
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        best_indices.append(best_idx)
    return best_indices


# Sample contour feature from image feature
def bilinear_interpolate_v2(img, coords):
    [n,c,h,w] = img.shape
    X = coords[:, :, 0] * w
    Y = coords[:, :, 1] * h

    X0 = torch.floor(X)
    X1 = X0 + 1
    Y0 = torch.floor(Y)
    Y1 = Y0 + 1

    W00 = (X1 - X) * (Y1 - Y)
    W01 = (X1 - X) * (Y - Y0)
    W10 = (X - X0) * (Y1 - Y)
    W11 = (X - X0) * (Y - Y0)

    X0 = torch.clamp(X0, 0, w-1).long()
    X1 = torch.clamp(X1, 0, w-1).long()
    Y0 = torch.clamp(Y0, 0, h-1).long()
    Y1 = torch.clamp(Y1, 0, h-1).long()

    I00 = X0 + Y0 * w
    I01 = X0 + Y1 * w
    I10 = X1 + Y0 * w
    I11 = X1 + Y1 * w

    flat_img = img.view(n,c,-1)
    P00 = torch.gather(flat_img, 2, I00.unsqueeze(1).repeat(1,c,1))
    P01 = torch.gather(flat_img, 2, I01.unsqueeze(1).repeat(1,c,1))
    P10 = torch.gather(flat_img, 2, I10.unsqueeze(1).repeat(1,c,1))
    P11 = torch.gather(flat_img, 2, I11.unsqueeze(1).repeat(1,c,1))

    return W00.unsqueeze(-1).repeat(1,1,c)*torch.transpose(P00,2,1) + W01.unsqueeze(-1).repeat(1, 1, c) * torch.transpose(P01, 2, 1) + \
            W10.unsqueeze(-1).repeat(1, 1, c) * torch.transpose(P10, 2, 1) + W11.unsqueeze(-1).repeat(1, 1, c) * torch.transpose(P11, 2, 1)


# Calculate and concatenate contour features
def get_ctr_feat_v2(vgg, img, ctr):
    [n, c, h, w] = img.shape
    img_feats = vgg(img)
    ctr_feat = torch.zeros((ctr.shape[0], ctr.shape[1], 0), dtype=torch.float32, device=img.device)
    for feat in img_feats:
        tmp_ctr_feat = bilinear_interpolate_v2(feat, ctr)
        ctr_feat = torch.cat((ctr_feat, tmp_ctr_feat), dim=2)
    return ctr_feat
