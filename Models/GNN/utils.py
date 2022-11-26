import torch


def offset_torch(coordinates, offsets):
    '''
    Create contours parallel to the center contour
    :param offsets: a list of offsets
    :return: return the parallel points of given coordinates
    '''
    parallel_points = []

    coordinates_temp1 = coordinates[:, :-1, :]
    coordinates_temp2 = coordinates[:, 1:, :]
    coor_mx = coordinates_temp2 - coordinates_temp1
    slop_mx = coor_mx[:, :, 1] / coor_mx[:, :, 0]
    pslop_mx = -1 / slop_mx
    mid_mx = (coordinates_temp1 + coordinates_temp2) / 2
    a = (pslop_mx > 0)
    b = (coordinates_temp1[:, :, 0] > coordinates_temp2[:, :, 0])
    c = a == b
    c = c.float()
    sign_mx = c * 2 - 1

    for offset in offsets:
        delta_x_mx = sign_mx * offset / ((1 + pslop_mx ** 2) ** 0.5)
        delta_y_mx = pslop_mx * delta_x_mx
        delta_x_mx = torch.cat((delta_x_mx, delta_x_mx[:, -1:]), dim=1)
        delta_y_mx = torch.cat((delta_y_mx, delta_y_mx[:, -1:]), dim=1)
        delta_x_mx = delta_x_mx.unsqueeze(2)
        delta_y_mx = delta_y_mx.unsqueeze(2)
        shift = torch.cat((delta_x_mx, delta_y_mx), dim=2)
        points = coordinates + shift
        parallel_points.append(points)
    return parallel_points


def interpolated_sum_multicontour(cnns, coords_multi, grid):
    '''
    Extract feature vectors from cnn feature map based on the given coordinates
    :param cnns: feature map to be extracted from
    :param coords_multi: coordinates to be used to extract features
    :param grid: int size of the feature map
    :return: feature vectors at each given coords_multi
    '''
    cnn_outs = []

    if not isinstance(grid, list):
        feat_h, feat_w = grid, grid
    elif len(grid) == 2:
        [feat_h, feat_w] = grid

    for coords in coords_multi:
        X = coords[:, :, 0]
        Y = coords[:, :, 1]

        Xs = X * feat_w
        X0 = torch.floor(Xs)
        X1 = X0 + 1

        Ys = Y * feat_h
        Y0 = torch.floor(Ys)
        Y1 = Y0 + 1

        w_00 = (X1 - Xs) * (Y1 - Ys)
        w_01 = (X1 - Xs) * (Ys - Y0)
        w_10 = (Xs - X0) * (Y1 - Ys)
        w_11 = (Xs - X0) * (Ys - Y0)

        X0 = torch.clamp(X0, 0, feat_w-1)
        X1 = torch.clamp(X1, 0, feat_w-1)
        Y0 = torch.clamp(Y0, 0, feat_h-1)
        Y1 = torch.clamp(Y1, 0, feat_h-1)

        N1_id = X0 + Y0 * feat_w
        N2_id = X0 + Y1 * feat_w
        N3_id = X1 + Y0 * feat_w
        N4_id = X1 + Y1 * feat_w

        M_00 = gather_feature(N1_id, cnns)
        M_01 = gather_feature(N2_id, cnns)
        M_10 = gather_feature(N3_id, cnns)
        M_11 = gather_feature(N4_id, cnns)
        cnn_out = w_00.unsqueeze(2) * M_00 + \
                  w_01.unsqueeze(2) * M_01 + \
                  w_10.unsqueeze(2) * M_10 + \
                  w_11.unsqueeze(2) * M_11

        cnn_outs.append(cnn_out)
    concat_features = torch.cat(cnn_outs, dim=2)
    return concat_features


def get_coors(scores, grid_size):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0]) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor(preds[:, :, 1] - 1) / scores.size(3) + 1

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask

    preds /= grid_size
    preds = torch.clamp(preds, 0, 1)

    return preds


def gather_feature(id, feature):
    feature_id = id.unsqueeze_(2).long().expand(id.size(0),
                                                id.size(1),
                                                feature.size(2)).detach()

    cnn_out = torch.gather(feature, 1, feature_id).float()

    return cnn_out