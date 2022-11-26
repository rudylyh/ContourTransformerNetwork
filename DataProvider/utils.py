import torch
import numpy as np
import json
from PIL import Image
import cv2
import os
from scipy.spatial.distance import cdist


def load_data(paths, index, type='handjoint'):
    '''
    Load datasets
    :param paths: the path of data
    :param index: index of data
    :param type: which dataset to load
    :return: loaded data
    '''

    path = paths[index]
    file_name = path.split('/')[-1].split('_vis')[0]
    ori_img = Image.open(path + '/' + file_name + '.png')
    json_file = path + '/' + file_name + '.json'

    with open(json_file, 'r') as f:
        points = json.load(f)

    if type == 'handjoint':
        target_control_lower = np.asarray(points['lower']['target_control'], dtype=np.float32)
        init_control_lower = np.asarray(points['lower']['init_control'], dtype=np.float32)
        target_control_upper = np.asarray(points['upper']['target_control'], dtype=np.float32)
        init_control_upper = np.asarray(points['upper']['init_control'], dtype=np.float32)
        return file_name, ori_img, target_control_lower, init_control_lower, target_control_upper, init_control_upper
    else:
        target_control = np.asarray(points['target_control'], dtype=np.float32)
        init_control = np.asarray(points['init_control'], dtype=np.float32)
        return file_name, ori_img, target_control, init_control


def rot_img_withcoords(image, img_size, target_controls, init_controls, degree):
    '''
    Rotate image as well as coordinates
    :param image: image
    :param img_size: image size
    :param target_controls: a list of target control points
    :param init_controls: a list of initial control points
    :param degree: the degree to rotate
    :return: rotated results
    '''

    rand = np.random.random()

    if rand < 0.8:
        angle = np.random.uniform(-degree, degree, 1) * 180
        target_controls_results = []
        init_controls_results = []
        x = image.rotate(angle, expand=True)
        org_center = (np.array(image.size[:2][::-1]) - 1) / 2.
        rot_center = (np.array([img_size, img_size]) - 1) / 2.
        image = x.crop(box=(x.size[0] / 2 - image.size[0] / 2,
                            x.size[1] / 2 - image.size[1] / 2,
                            x.size[0] / 2 + image.size[0] / 2,
                            x.size[1] / 2 + image.size[1] / 2))
        # for each_place in target_controls:
        #     each_place *= img_size
        #     result = []
        #     for each in each_place:
        #         org = each - org_center
        #         a = np.deg2rad(angle)
        #         new = np.array([org[0] * np.cos(a) + org[1] * np.sin(a),
        #                         -org[0] * np.sin(a) + org[1] * np.cos(a)])
        #         result.append(new.transpose())
        #     result = np.concatenate(result, axis=0) + rot_center
        #     result /= img_size
        #     target_controls_results.append(result.astype(np.float32))
        # for each_place in init_controls:
        #     each_place *= img_size
        #     result = []
        #     for each in each_place:
        #         org = each - org_center
        #         a = np.deg2rad(angle)
        #         new = np.array([org[0] * np.cos(a) + org[1] * np.sin(a),
        #                         -org[0] * np.sin(a) + org[1] * np.cos(a)])
        #         result.append(new.transpose())
        #     result = np.concatenate(result, axis=0) + rot_center
        #     result /= img_size
        #     init_controls_results.append(result.astype(np.float32))
    else:
        image = image
        # target_controls_results = target_controls
        # init_controls_results = init_controls

    return image, target_controls, init_controls


def flip_leftright(image, target_controls, init_controls, flip):
    '''
    Flip initial image as well as controls left and right
    :param image: image to flip
    :param img_size: image size
    :param target_controls: target control points
    :param init_controls: initial control points
    :return: flipped results
    '''
    # rand = np.random.random()
    # if rand < 0.5:
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        for i, control in enumerate(target_controls):
            control[:, 0] = 1 - control[:, 0]
            target_controls[i] = control
        for i, control in enumerate(init_controls):
            control[:, 0] = 1 - control[:, 0]
            init_controls[i] = control
    return image, target_controls, init_controls


def flip_leftright_pelvis(image, target_controls, init_controls):
    '''
    Flip initial image as well as controls left and right.
    Especially designed for pelvis for keeping the control point semantic information unchanged.
    E.g. the flipped PR1 is considered as PL1.
    :param image: image to flip
    :param img_size: image size
    :param target_controls: target control points
    :param init_controls: initial control points
    :return: flipped results
    '''

    rand = np.random.random()
    if rand < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        for i, control in enumerate(target_controls):
            control[:, 0] = 1 - control[:, 0]
            temp = control[0:7].copy()
            control[0:7] = control[9:]
            control[9:] = temp
            target_controls[i] = control

        for i, control in enumerate(init_controls):
            control[:, 0] = 1 - control[:, 0]
            temp = control[0:7].copy()
            control[0:7] = control[9:]
            control[9:] = temp
            init_controls[i] = control
    return image, target_controls, init_controls


def normalize_controls(target_controls, init_controls):
    '''
    Normalized control points
    :param target_controls: a list of target control points
    :param init_controls: a list of initial control points
    :param img_size: image size used to normalized control points
    :return: normalized control points
    '''

    for i, control in enumerate(target_controls):
        # control /= img_size
        control = control.clip(min=0).clip(max=1)
        target_controls[i] = control
    for i, control in enumerate(init_controls):
        # control /= img_size
        control = control.clip(min=0).clip(max=1)
        init_controls[i] = control
    return target_controls, init_controls


def shift_init(init_controls, img_size):
    '''
    Shift initial control points
    :param init_controls: initial control points
    :return: shifted initial control points
    '''

    rand = np.random.random()
    if rand < 0.5:
        shift_lower_x = np.random.uniform(low=-1, high=1, size=(1,)) * (15/img_size)
        shift_lower_y = np.random.uniform(low=-1, high=1, size=(1,)) * (15/img_size)
        for i, init_control in enumerate(init_controls):
            init_control[:, 0] += shift_lower_x
            init_control[:, 1] += shift_lower_y
            init_controls[i] = init_control
    return init_controls


def scale_init(init_controls):
    '''
    Scale initial control points
    :param init_controls: initial control points
    :return: scaled initial control points
    '''

    rand = np.random.random()
    if rand < 0.5:
        rand_scale = np.random.uniform(0.8, 1.2)
        for i, init_control in enumerate(init_controls):
            o = (init_control.min(axis=0) + init_control.max(axis=0)) * .5
            init_control = o * (1 - rand_scale) + init_control * rand_scale
            init_controls[i] = init_control
    return init_controls


def generate_rois(h0, w0, h1, w1, h, w):
    roi = np.zeros((9, 4))
    roi[0, 0] = w0
    roi[0, 1] = h0
    roi[0, 2] = w1
    roi[0, 3] = h1

    roi[1, 0] = 0
    roi[1, 1] = 0
    roi[1, 2] = max(0, w0-1)
    roi[1, 3] = max(0, h0-1)

    roi[2, 0] = 0
    roi[2, 1] = h0
    roi[2, 2] = max(0, w0-1)
    roi[2, 3] = max(0, h1-1)

    roi[3, 0] = 0
    roi[3, 1] = h1
    roi[3, 2] = max(0, w0-1)
    roi[3, 3] = h

    roi[4, 0] = w0
    roi[4, 1] = max(0, h1+1)
    roi[4, 2] = w1
    roi[4, 3] = h

    roi[5, 0] = min(w1+1, w)
    roi[5, 1] = h1
    roi[5, 2] = w
    roi[5, 3] = h

    roi[6, 0] = min(w1+1, w)
    roi[6, 1] = h0
    roi[6, 2] = w
    roi[6, 3] = max(h1-1, 0)

    roi[7, 0] = min(w1+1, w)
    roi[7, 1] = 0
    roi[7, 2] = w
    roi[7, 3] = max(h0-1, 0)

    roi[8, 0] = w0
    roi[8, 1] = 0
    roi[8, 2] = w1
    roi[8, 3] = max(h0-1, 0)

    return roi


def generate_heatmap(target_controls, grid_size, sigma, label_type='Gaussian', cp_num=16):
    '''
    Generate Gaussian heatmaps centered on the given target control points
    :param target_controls: target control points
    :param grid_size: the target heatmap dimension
    :param sigma: sigma for gaussian
    :param label_type: which type of distribution to use
    :param cp_num: number of control points (equal to number of heatmaps)
    :return: generated heatmaps
    '''

    heatmaps = []
    for each_target in target_controls:
        img = np.zeros((cp_num, grid_size, grid_size)).astype(np.float32)
        for i, each_point in enumerate(each_target):
            pt = each_point * grid_size
            # Check that any part of the gaussian is in-bounds
            tmp_size = sigma * 3
            ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
            br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
            if (ul[0] >= grid_size or ul[1] >= grid_size or
                    br[0] < 0 or br[1] < 0):
                # If not, just return the image as is
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            if label_type == 'Gaussian':
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
            else:
                g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], grid_size) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], grid_size) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], grid_size)
            img_y = max(0, ul[1]), min(br[1], grid_size)

            img[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        heatmaps.append(img)
    return heatmaps


def create_adjacency_matrix_pelvis(batch_size, n_nodes):
    '''
    A predefined adjacency matrix for pelvis dataset
    :param batch_size: batch size
    :param n_nodes: number of nodes (here by default 16 landmarks)
    :return: adjacency matrix [batch_size, n_nodes, n_nodes]
    '''

    a = np.zeros([batch_size, n_nodes, n_nodes])
    a[0][0][1] = 1
    a[0][0][2] = 1
    a[0][1][2] = 1
    a[0][1][3] = 1
    a[0][2][3] = 1
    a[0][3][4] = 1
    a[0][3][5] = 1
    a[0][3][7] = 1
    a[0][4][5] = 1
    a[0][4][6] = 1
    a[0][4][8] = 1
    a[0][5][6] = 1
    a[0][7][8] = 1
    a[0][7][12] = 1
    a[0][8][13] = 1
    a[0][9][10] = 1
    a[0][9][11] = 1
    a[0][10][11] = 1
    a[0][10][12] = 1
    a[0][11][12] = 1
    a[0][12][13] = 1
    a[0][12][14] = 1
    a[0][13][14] = 1
    a[0][13][15] = 1
    a[0][14][15] = 1
    i_lower = np.tril_indices(16, -1)
    a[0][i_lower] = a[0].T[i_lower]
    return a.astype(np.float32)


def create_adjacency_matrix_wholehand(batch_size, n_nodes):
    '''
    A predefined adjacency matrix for pelvis dataset
    :param batch_size: batch size
    :param n_nodes: number of nodes (here by default 16 landmarks)
    :return: adjacency matrix [batch_size, n_nodes, n_nodes]
    '''

    a = np.zeros([batch_size, n_nodes, n_nodes])
    a[0][0][1] = 1
    a[0][1][2] = 1
    a[0][2][3] = 1

    a[0][4][5] = 1
    a[0][5][6] = 1
    a[0][6][7] = 1
    a[0][7][8] = 1

    a[0][9][10] = 1
    a[0][10][11] = 1
    a[0][11][12] = 1
    a[0][12][13] = 1

    a[0][14][15] = 1
    a[0][15][16] = 1
    a[0][16][17] = 1
    a[0][17][18] = 1

    a[0][19][20] = 1
    a[0][20][21] = 1
    a[0][21][22] = 1
    a[0][22][23] = 1

    a[0][24][25] = 1
    a[0][25][26] = 1
    a[0][26][27] = 1
    a[0][27][28] = 1
    a[0][28][29] = 1

    a[0][23][24] = 1
    a[0][3][29] = 1

    a[0][3][8] = 1
    a[0][8][13] = 1
    a[0][13][18] = 1
    a[0][18][23] = 1

    i_lower = np.tril_indices(30, -1)
    a[0][i_lower] = a[0].T[i_lower]
    return a.astype(np.float32)


def create_adjacency_matrix_aasce(batch_size, n_nodes):
    '''
    A predefined adjacency matrix for aasce dataset
    :param batch_size: batch size
    :param n_nodes: number of nodes (here by default 68 landmarks)
    :return: adjacency matrix [batch_size, n_nodes, n_nodes]
    '''

    a = np.zeros([batch_size, n_nodes, n_nodes])

    for t in range(batch_size):
        for i in range(n_nodes):
            if i % 2 == 0 or i == 0:
                if i + 1 < 68:
                    a[t][i][(i + 1)] = 1
                if i + 2 < 68:
                    a[t][i][(i + 2)] = 1
            if i % 2 != 0 and i+2 < 68:
                a[t][i][(i + 2)] = 1
    i_lower = np.tril_indices(68, -1)
    a[0][i_lower] = a[0].T[i_lower]
    return a.astype(np.float32)


def create_adjacency_matrix(batch_size, n_neighbor, n_nodes, is_circle=False):
    '''
    Create curve-structured adjacenry matrix
    :param batch_size: batch size
    :param n_neighbor: number of neighbors to be considered
    :param n_nodes: number of nodes on the curve
    :return: adjacency matrix [batch_size, n_nodes, n_nodes]
    '''

    a = np.zeros([batch_size, n_nodes, n_nodes])

    for t in range(batch_size):
        for i in range(n_nodes):
            for j in range(-n_neighbor // 2, n_neighbor // 2 + 1):
                if j != 0:
                    # Break start-end connectivity
                    if is_circle==False:
                        if i+j < 0 or i+j >= n_nodes:
                            continue
                    a[t][i][(i + j) % n_nodes] = 1
                    a[t][(i + j) % n_nodes][i] = 1
    return a.astype(np.float32)


def prepare_gcn_component_pelvis(n_nodes):
    ''''
    Prepare adjacency matrix for pelvis dataset
    :param n_nodes: number of nodes
    :return: adjacency matrix [n_nodes, no_nodes]
    '''

    adj_matrix = create_adjacency_matrix_pelvis(1, n_nodes).squeeze()
    return {'adj_matrix': torch.Tensor(adj_matrix)}


def prepare_gcn_component_wholehand(n_nodes):
    ''''
    Prepare adjacency matrix for pelvis dataset
    :param n_nodes: number of nodes
    :return: adjacency matrix [n_nodes, no_nodes]
    '''

    adj_matrix = create_adjacency_matrix_wholehand(1, n_nodes).squeeze()
    return {'adj_matrix': torch.Tensor(adj_matrix)}


def prepare_gcn_component_aasce(n_nodes):
    ''''
    Prepare adjacency matrix for pelvis dataset
    :param n_nodes: number of nodes
    :return: adjacency matrix [n_nodes, no_nodes]
    '''

    adj_matrix = create_adjacency_matrix_aasce(1, n_nodes).squeeze()
    return {'adj_matrix': torch.Tensor(adj_matrix)}


def prepare_gcn_component_knuckle(n_neighbor, n_nodes):
    '''
    Prepare adjacency matrix for knuckle dataset
    :param n_nodes: number of nodes
    :param n_neighbor: number of neighbors to be used
    :return: adjacency matrix [n_nodes, no_nodes]
    '''

    adj_matrix = create_adjacency_matrix(1, n_neighbor, n_nodes, is_circle=True).squeeze()
    return {'adj_matrix': torch.Tensor(adj_matrix)}


def prepare_gcn_component_lung(n_neighbor, n_nodes):
    '''
    Prepare adjacency matrix for knuckle dataset
    :param n_nodes: number of nodes
    :param n_neighbor: number of neighbors to be used
    :return: adjacency matrix [n_nodes, no_nodes]
    '''

    adj_matrix = create_adjacency_matrix(1, n_neighbor, n_nodes, is_circle=True).squeeze()
    return {'adj_matrix': torch.Tensor(adj_matrix)}


def prepare_gcn_component(n_neighbor, n_nodes):
    '''
    Prepare adjacency matrix for curve-structured dataset
    :param n_nodes: number of nodes
    :param n_neighbor: number of neighbors to be used
    :return: adjacency matrix [n_nodes, no_nodes]
    '''

    adj_matrix = create_adjacency_matrix(1, n_neighbor, n_nodes).squeeze()
    return {'adj_matrix': torch.Tensor(adj_matrix)}


def sample_point(old_pts, new_num):
    old_num = old_pts.shape[0]
    old_pts = np.concatenate((old_pts, np.expand_dims(old_pts[0], axis=0)))
    old_x, old_y = old_pts[:,0], old_pts[:,1]

    old_idx = np.arange(0, old_num + 1)
    interval = float(old_num) / new_num
    sampled_idx = np.linspace(0, old_num-interval, new_num)

    new_x = np.interp(sampled_idx, old_idx, old_x)
    new_y = np.interp(sampled_idx, old_idx, old_y)
    new_pts = np.stack((new_x, new_y), axis=1)

    return new_pts


def sample_arc_point(old_pts, new_num):
    """
    Sample points from a given sequence of points
    :param old_pts: input sequence of points, numpy.ndarray, Nx2
    :param new_num: desired number of points, int
    :return: output sequence of points, numpy.ndarray, Nx2
    """
    old_num = old_pts.shape[0]
    # old_pts = np.concatenate((old_pts, np.expand_dims(old_pts[0], axis=0)))
    old_x, old_y = old_pts[:,0], old_pts[:,1]
    dx = old_x[1:] - old_x[:-1]
    dy = old_y[1:] - old_y[:-1]
    dlength = (dx ** 2 + dy ** 2) ** 0.5
    arc_length = np.cumsum(dlength)
    arc_length = np.concatenate((np.zeros((1,)), arc_length))
    sampled_idx = np.linspace(0, arc_length[-1], new_num)
    old_idx = np.arange(0, old_num + 1)
    new_x = np.interp(sampled_idx, arc_length, old_x)
    new_y = np.interp(sampled_idx, arc_length, old_y)
    new_pts = np.stack((new_x, new_y), axis=1)
    return new_pts


def show_multi_ctrs(img, ctr1, ctr2, ctr3, ctr4):
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    ctr1 = np.round(ctr1).astype(int)
    ctr2 = np.round(ctr2).astype(int)
    ctr3 = np.round(ctr3).astype(int)
    ctr4 = np.round(ctr4).astype(int)
    for p in ctr1:
        vis_img[p[1], p[0]] = [0, 0, 255]
    for p in ctr2:
        vis_img[p[1], p[0]] = [0, 255, 0]
    for p in ctr3:
        vis_img[p[1], p[0]] = [0, 255, 255]
    for p in ctr4:
        vis_img[p[1], p[0]] = [255, 0, 0]
    cv2.imshow('1', vis_img)
    cv2.waitKey()


def get_gradient_magnitude(img):
    "Get the magnitude of gradient for given image"
    img = np.array(img, dtype=np.float32)
    fx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    fy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)
    # fx = gaussian_filter(img, sigma=3, order=(0,1))
    # fy = gaussian_filter(img, sigma=3, order=(1,0))
    grad_mag = np.sqrt(fx * fx + fy * fy)
    # grad_mag = cv2.addWeighted(cv2.convertScaleAbs(fx), 0.5, cv2.convertScaleAbs(fy), 0.5, 0)
    min_grad, max_grad = np.min(grad_mag), np.max(grad_mag)
    grad_mag = (grad_mag - min_grad) / (max_grad - min_grad)
    return grad_mag


def get_smooth_img(img):
    smooth_img = cv2.medianBlur(img, 11)
    min_val, max_val = np.min(smooth_img), np.max(smooth_img)
    smooth_img = (smooth_img.astype(np.float32) - min_val) / (max_val - min_val)
    smooth_img = (255*smooth_img).astype(np.uint8)
    return smooth_img


def adjust_contrast(img, lower_percent_thre, higher_percent_thre, max_grey=256):
    hist, bins = np.histogram(img.ravel(), max_grey, [0, max_grey])
    accum_hist = np.cumsum(hist)
    low_grey_thre, high_grey_thre = np.min(img), np.max(img)
    for i in range(0, len(hist)):
        if accum_hist[i] >= lower_percent_thre * img.size:
            low_grey_thre = i
            break
    for i in range(0, len(hist)):
        if accum_hist[i] >= higher_percent_thre * img.size:
            high_grey_thre = i
            break
    img[img < low_grey_thre] = low_grey_thre
    img[img > high_grey_thre] = high_grey_thre
    img = img.astype(np.float32)
    img = (img - low_grey_thre) / (high_grey_thre - low_grey_thre)
    return img


def adjust_grad_contrast(grad_mag, max_thre):
    img = (255 * grad_mag).astype(np.uint8)
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    accum_hist = np.cumsum(hist)
    low_grey_thre, high_grey_thre = np.min(img), np.max(img)
    for i in range(0, len(hist)):
        if accum_hist[i] >= max_thre * img.size:
            high_grey_thre = i
            break
    img[img > high_grey_thre] = 0
    img = img.astype(np.float32) / high_grey_thre
    return img


# Find wrong segments in a prediction contour
def get_partial_target(gt, pred, is_circle=True):
    thre = 3 # distance threshold
    min_len = 5 # length threshold
    min_gap = 10 # merge two segments if too close

    pt_num = gt.shape[0]
    gt_to_pred_dist = cdist(gt, pred, 'euclidean').min(axis=1)
    wrong_pt_idx = np.argwhere(gt_to_pred_dist > thre).squeeze()
    if len(wrong_pt_idx.shape) == 0 or wrong_pt_idx.shape[0] == 0:
        return np.array([])

    # remove too small segments and bridge very close segments
    split_wrong_pt_idx = list()
    tmp_idx_list = list()
    for i in range(0, wrong_pt_idx.shape[0]-1):
        tmp_idx_list.append(wrong_pt_idx[i])
        if wrong_pt_idx[i]+1 != wrong_pt_idx[i+1] :
            if wrong_pt_idx[i] + min_gap >= wrong_pt_idx[i+1]:
                tmp_idx_list += list(np.arange(wrong_pt_idx[i]+1, wrong_pt_idx[i+1]))
            else:
                if len(tmp_idx_list) >= min_len:
                    split_wrong_pt_idx.append(np.array(tmp_idx_list).squeeze())
                tmp_idx_list = list()
    tmp_idx_list.append(wrong_pt_idx[-1]) # For the last segment
    if len(tmp_idx_list) >= min_len:
        split_wrong_pt_idx.append(np.array(tmp_idx_list).squeeze())

    # concatenate the last and the first segment, if close
    if is_circle and len(split_wrong_pt_idx) >= 2 and split_wrong_pt_idx[-1][-1]+min_gap >= split_wrong_pt_idx[0][0] + pt_num:
        split_wrong_pt_idx[0] = np.concatenate((np.arange(split_wrong_pt_idx[-1][0], pt_num), np.arange(0, split_wrong_pt_idx[0][-1]+1)))
        split_wrong_pt_idx.pop()

    # use -1 as the delimiter of each gt segment
    all_gt_segs = list()
    for idx_list in split_wrong_pt_idx:
        for idx in idx_list:
            all_gt_segs.append(gt[idx].squeeze())
        all_gt_segs.append(np.array([-1.0,-1.0]))
    return np.round(np.array(all_gt_segs))
