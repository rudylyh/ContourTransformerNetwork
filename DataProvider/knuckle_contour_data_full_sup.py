import os
import numpy as np
from scipy.ndimage import laplace
import json
from PIL import Image
import cv2
from torch.utils import data
import DataProvider.utils as DataUtils
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
from skimage import filters
from scipy import ndimage
import torch


class SelfDataset(data.Dataset):
    def __init__(self, mode, opts):
        self.mode = mode
        self.opts = opts
        data_dict = json.load(open(opts['data_list']))
        self.root = data_dict['root_dir']
        self.data_list = data_dict[self.mode]
        # self.data_list = self.data_list[0:500]
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # Get image
        img_name = self.data_list[index]
        ori_img_path = os.path.join(self.root, img_name, img_name+'.png')
        ori_img = cv2.imread(ori_img_path, 0)
        smooth_img = DataUtils.get_smooth_img(ori_img)
        grad_mag = DataUtils.get_gradient_magnitude(smooth_img)

        # Get init contour
        [h,w] = self.opts['img_size']
        init_ctr_path = os.path.join(self.root, img_name, img_name + '_init.json')
        # init_ctr_path = '/home/yuhang/workspace/ctr_exp/manual_ctr/knuckle_mean_ctr.json'
        init_ctr_json = json.load(open(init_ctr_path))
        init_control = np.asarray(init_ctr_json['init_control'], dtype=np.float32)
        init_control = DataUtils.sample_arc_point(init_control, self.opts['cp_num'])
        # init_control = DataUtils.get_circle_init(ori_img)
        # init_control = sample_point_v2(torch.Tensor(init_control).unsqueeze(0), 200, 1000)
        init_control = (init_control / [w,h]).astype(np.float32) # convert to float32 again, or else it will be float64

        # Get target contour, if exists
        target_ctr_path = os.path.join(self.root, img_name, img_name + '_target.json')
        target_control = np.zeros_like(init_control, dtype=np.float32)
        if os.path.exists(target_ctr_path):
            target_ctr_json = json.load(open(target_ctr_path))
            full_gt_ctr = np.asarray(target_ctr_json['target_control'], dtype=np.float32)
            target_control = DataUtils.sample_arc_point(full_gt_ctr, self.opts['cp_num'])
            target_control = (target_control / [w,h]).astype(np.float32)

        # vis_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2RGB)
        # ctr1 = (init_control * [w, h])
        # ctr2 = (target_control * [w, h])
        # ctr1 = np.round(ctr1).astype(int)
        # ctr2 = np.round(ctr2).astype(int)
        # for p in ctr1:
        #     vis_img[p[1], p[0]] = [0, 0, 255]
        # for p in ctr2:
        #     vis_img[p[1], p[0]] = [0, 255, 0]
        # cv2.imshow('1', vis_img)
        # cv2.waitKey()

        gcn_img = np.array(smooth_img)
        gcn_img = np.expand_dims(gcn_img, axis=2)
        gcn_img = np.repeat(gcn_img, 3, axis=2)
        gcn_img = gcn_img.astype(np.float32) / 255.0
        gcn_img = gcn_img.transpose(2, 0, 1)

        # Gray to RGB
        ori_img = np.expand_dims(ori_img, axis=2)
        ori_img = np.repeat(ori_img, 3, axis=2).transpose((2, 0, 1))

        # Prepare GCN components
        gcn_component = DataUtils.prepare_gcn_component_knuckle(self.opts['n_neighbor'], self.opts['cp_num'])
        # gcn_component['adj_matrix'] = torch.eye(init_control.shape[0])
        # gcn_component['adj_matrix'] = torch.ones(init_control.shape[0], init_control.shape[0])

        data_in = {'file_name': img_name,
                   'ori_img': ori_img,
                   'smooth_img': smooth_img,
                   'gcn_img': gcn_img,
                   'grad_mag': grad_mag,
                   'init_control': [init_control],
                   'target_control': [target_control],
                   'gcn_component': [gcn_component]}

        return data_in
