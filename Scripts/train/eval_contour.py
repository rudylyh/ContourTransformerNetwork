import torch
import json
import os
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from collections import defaultdict
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
sys.path.append('.')
from Scripts import utils as trainUtil
from Models import model_gnn
from Evaluation import losses, metrics
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from Models.sync_batchnorm import convert_model
import cv2
from Models.Encoder.vgg_justin import Vgg16


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    return args


class Eval(object):
    def __init__(self, args):
        self.opts = json.load(open(args.exp, 'r'))
        self.global_step = 0
        self.epoch = 0
        self.device = torch.device("cuda:{}".format(self.opts['gpus'][0]))

        self.train_loader = trainUtil.get_data_loaders(self.opts, mode='train', shuffle=False)
        self.val_loader = trainUtil.get_data_loaders(self.opts, mode='val', shuffle=False)
        self.test_loader = trainUtil.get_data_loaders(self.opts, mode='test', shuffle=False)
        self.model = model_gnn.PolyGNN(gcn_out_dim=self.opts['gcn_out_dim'],
                                       gcn_steps=self.opts['gcn_steps'],
                                       nInputChannels=self.opts['nInputChannels'],
                                       opts=self.opts)

        self.model = convert_model(self.model)
        self.model = torch.nn.DataParallel(self.model.to(self.device), device_ids=self.opts['gpus'])

        self.resume(args.resume)
        self.ckp_path = os.path.dirname(args.resume)

        self.sample_data = dict()
        for step, data in enumerate(self.train_loader):
            for i in range(0, len(data['file_name'])):
                if data['file_name'][i] == self.opts['one_shot_sample']:
                    self.sample_data['file_name'] = data['file_name'][i]
                    self.sample_data['gcn_img'] = data['gcn_img'][i].to(self.device)
                    self.sample_data['gt_ctr'] = data['target_control'][0][i].to(self.device)
                    break

        self.vgg = Vgg16().type(torch.cuda.FloatTensor).to(self.device)


    def resume(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict['state_dict'])

    ### For the validation set
    def validate(self):
        print('Validating')
        self.model.eval()
        with torch.no_grad():
            val_avg_dist, val_max_dist, val_IOU, num_imgs = 0.0, 0.0, 0.0, 0
            [res_rows, res_cols] = self.opts['val_img_num']
            [h, w] = self.opts['img_size']
            pred_result_img = Image.new('RGB', (res_cols * w, res_rows * h))
            for step, data in enumerate(tqdm(self.val_loader)):
                img = data['gcn_img'].to(self.device)
                grad_mag = data['grad_mag'].to(self.device)
                target_control = data['target_control'][0].to(self.device)
                # init_control = data['init_control'][0].to(self.device)
                init_control = self.sample_data['gt_ctr'].repeat(img.shape[0], 1, 1)
                gcn_component = {'adj_matrix': data['gcn_component'][0]['adj_matrix'].to(self.device)}
                output = self.model(img, [init_control], [gcn_component])
                num_imgs += img.shape[0]
                pred_control = output['pred_coor'][0][-1]

                for k in range(img.shape[0]):
                    # Compute Hausdorff distance
                    pred = pred_control[k].data.cpu().numpy() * [w, h]
                    target = target_control[k].data.cpu().numpy() * [w, h]
                    init = init_control[k].data.cpu().numpy() * [w, h]

                    avg_dist, max_dist = metrics.hausdorff_dist(pred, target)
                    IOU = metrics.get_ctr_iou(np.round(pred).astype(np.int32), np.round(target).astype(np.int32), h, w)
                    val_avg_dist += avg_dist
                    val_max_dist += max_dist
                    val_IOU += IOU

                    # Draw figures
                    file_name = data['file_name'][k]
                    img = (data['ori_img'][k, ...].cpu().numpy()).astype(np.uint8)
                    img = np.transpose(img, [1, 2, 0])
                    PIL_img = Image.fromarray(img)
                    poly_img = PIL_img.copy()
                    draw_gt = ImageDraw.Draw(poly_img)

                    draw_gt.point([tuple(x) for x in target.astype(np.int32)], fill="green")
                    draw_gt.point([tuple(x) for x in init.astype(np.int32)], fill="red")
                    draw_gt.point([tuple(x) for x in pred.astype(np.int32)], fill="yellow")
                    draw_gt.text((0, 0), file_name, (0, 255, 0), font=ImageFont.load_default())
                    draw_gt.text((0, 12), "IOU: {0:.3f}".format(IOU), (0, 255, 0), font=ImageFont.load_default())
                    draw_gt.text((0, 24), "avg_dist: {0:.3f}".format(avg_dist), (0, 255, 0), font=ImageFont.load_default())
                    draw_gt.text((0, 36), "max_dist: {0:.3f}".format(max_dist), (0, 255, 0), font=ImageFont.load_default())

                    img_idx = step * self.opts['dataset']['val']['batch_size'] + k
                    row = img_idx // res_cols
                    col = img_idx % res_cols
                    pred_result_img.paste(poly_img, (col * w, row * h))

            print('\r[VAL] avg_dist: {0:.4f}, max_dist: {1:.4f}, IOU: {2:.4f}'.format(val_avg_dist/num_imgs, val_max_dist/num_imgs, val_IOU/num_imgs))
            pred_result_img.save(os.path.join(self.ckp_path, 'val.png'))
            pred_result_img.show()

    ### For the test set
    def test(self):
        print('Testing')
        self.model.eval()
        with torch.no_grad():
            val_avg_dist, val_max_dist, val_IOU, num_imgs = 0.0, 0.0, 0.0, 0
            [res_rows, res_cols] = self.opts['val_img_num']
            [h, w] = self.opts['img_size']
            pred_result_img = Image.new('RGB', (res_cols * w, res_rows * h))
            for step, data in enumerate(tqdm(self.test_loader)):
                img = data['gcn_img'].to(self.device)
                grad_mag = data['grad_mag'].to(self.device)
                target_control = data['target_control'][0].to(self.device)
                # init_control = data['init_control'][0].to(self.device)
                init_control = self.sample_data['gt_ctr'].repeat(img.shape[0], 1, 1)
                gcn_component = {'adj_matrix': data['gcn_component'][0]['adj_matrix'].to(self.device)}
                output = self.model(img, [init_control], [gcn_component])
                num_imgs += img.shape[0]

                pred_control = output['pred_coor'][0][-1]

                for k in range(img.shape[0]):
                    # Compute Hausdorff distance
                    pred = pred_control[k].data.cpu().numpy() * [w, h]
                    target = target_control[k].data.cpu().numpy() * [w, h]
                    init = init_control[k].data.cpu().numpy() * [w, h]

                    file_name = data['file_name'][k]
                    # with open(os.path.join(self.opts['test_write_dir'], file_name, file_name +'_pred_' + os.path.basename(self.ckp_path) + '.json'), 'w') as out_file:
                    #     json.dump({'pred_control': pred.tolist()}, out_file)

                    avg_dist, max_dist = metrics.hausdorff_dist(pred, target)
                    IOU = metrics.get_ctr_iou(np.round(pred).astype(np.int32), np.round(target).astype(np.int32), h, w)

                    val_avg_dist += avg_dist
                    val_max_dist += max_dist
                    val_IOU += IOU

                    # Draw figures
                    file_name = data['file_name'][k]
                    img = (data['ori_img'][k, ...].cpu().numpy()).astype(np.uint8)
                    img = np.transpose(img, [1, 2, 0])
                    PIL_img = Image.fromarray(img)
                    poly_img = PIL_img.copy()
                    draw_gt = ImageDraw.Draw(poly_img)

                    draw_gt.point([tuple(x) for x in target.astype(np.int32)], fill="green")
                    draw_gt.point([tuple(x) for x in init.astype(np.int32)], fill="red")
                    draw_gt.point([tuple(x) for x in pred.astype(np.int32)], fill="yellow")
                    draw_gt.text((0, 0), file_name, (0, 255, 0), font=ImageFont.load_default())
                    draw_gt.text((0, 12), "IOU: {0:.3f}".format(IOU), (0, 255, 0), font=ImageFont.load_default())
                    draw_gt.text((0, 24), "avg_dist: {0:.3f}".format(avg_dist), (0, 255, 0), font=ImageFont.load_default())
                    draw_gt.text((0, 36), "max_dist: {0:.3f}".format(max_dist), (0, 255, 0), font=ImageFont.load_default())

                    img_idx = step * self.opts['dataset']['test']['batch_size'] + k
                    row = img_idx // res_cols
                    col = img_idx % res_cols
                    pred_result_img.paste(poly_img, (col * w, row * h))

            print('\r[VAL] avg_dist: {0:.4f}, max_dist: {1:.4f}, IOU: {2:.7f}'.format(val_avg_dist/num_imgs, val_max_dist/num_imgs, val_IOU/num_imgs))
            save_name = 'test_%.4f_%.4f_%.4f.png' % (val_avg_dist/num_imgs, val_max_dist/num_imgs, val_IOU/num_imgs)
            # pred_result_img.save(os.path.join(self.ckp_path, save_name))
            # pred_result_img.save(os.path.join('/WD1/paii_internship/workspace/tmi_rebuttal/disturb_roi_v2', save_name))
            pred_result_img.show()

    # Write the predicted contours of training images and rank them by Hausdorff distance
    def generate_pred_for_train(self):
        self.model.eval()
        with torch.no_grad():
            val_avg_dist, val_max_dist, val_IOU, num_imgs = 0.0, 0.0, 0.0, 0
            [res_rows, res_cols] = self.opts['val_img_num']
            [h, w] = self.opts['img_size']
            pred_result_img = Image.new('RGB', (res_cols * w, res_rows * h))
            max_dist_dict = dict()
            for step, data in enumerate(tqdm(self.train_loader)):
                img = data['gcn_img'].to(self.device)
                grad_mag = data['grad_mag'].to(self.device)
                target_control = data['target_control'][0].to(self.device)
                # init_control = data['init_control'][0].to(self.device)
                init_control = self.sample_data['gt_ctr'].repeat(img.shape[0], 1, 1)
                gcn_component = {'adj_matrix': data['gcn_component'][0]['adj_matrix'].to(self.device)}
                output = self.model(img, [init_control], [gcn_component])
                num_imgs += img.shape[0]
                pred_control = output['pred_coor'][0][-1]

                for k in range(img.shape[0]):
                    # Compute Hausdorff distance
                    pred = pred_control[k].data.cpu().numpy() * [w, h]
                    target = target_control[k].data.cpu().numpy() * [w, h]
                    init = init_control[k].data.cpu().numpy() * [w, h]

                    # Save prediction json
                    file_name = data['file_name'][k]
                    # print(file_name)
                    # with open(os.path.join(self.opts['test_write_dir'], file_name, file_name +'_pred_' + os.path.basename(self.ckp_path) + '.json'), 'w') as out_file:
                    #     json.dump({'pred_control': pred.tolist()}, out_file)

                    avg_dist, max_dist = metrics.hausdorff_dist(pred, target)
                    IOU = metrics.get_ctr_iou(np.round(pred).astype(np.int32), np.round(target).astype(np.int32), h, w)
                    val_avg_dist += avg_dist
                    val_max_dist += max_dist
                    val_IOU += IOU
                    max_dist_dict[file_name] = max_dist

                    # Draw figures
                    img = (data['ori_img'][k, ...].cpu().numpy()).astype(np.uint8)
                    img = np.transpose(img, [1, 2, 0])
                    PIL_img = Image.fromarray(img)
                    poly_img = PIL_img.copy()
                    draw_gt = ImageDraw.Draw(poly_img)

                    draw_gt.point([tuple(x) for x in target.astype(np.int32)], fill="green")
                    draw_gt.point([tuple(x) for x in init.astype(np.int32)], fill="red")
                    draw_gt.point([tuple(x) for x in pred.astype(np.int32)], fill="yellow")
                    draw_gt.text((0, 0), file_name, (0, 255, 0), font=ImageFont.load_default())
                    draw_gt.text((0, 12), "IOU: {0:.3f}".format(IOU), (0, 255, 0), font=ImageFont.load_default())
                    draw_gt.text((0, 24), "avg_dist: {0:.3f}".format(avg_dist), (0, 255, 0), font=ImageFont.load_default())
                    draw_gt.text((0, 36), "max_dist: {0:.3f}".format(max_dist), (0, 255, 0), font=ImageFont.load_default())

                    img_idx = step * self.opts['dataset']['train']['batch_size'] + k
                    row = img_idx // res_cols
                    col = img_idx % res_cols
                    pred_result_img.paste(poly_img, (col * w, row * h))

            # print('\r[VAL] avg_dist: {0:.4f}, max_dist: {1:.4f}, IOU: {2:.4f}'.format(val_avg_dist/num_imgs, val_max_dist/num_imgs, val_IOU/num_imgs))
            # save_name = 'train_%.4f_%.4f_%.4f.png' % (val_avg_dist/num_imgs, val_max_dist/num_imgs, val_IOU/num_imgs)
            # pred_result_img.save(os.path.join(self.ckp_path, save_name))
            # # pred_result_img.show()
            #
            # sorted_dist_dict = sorted(max_dist_dict, key=max_dist_dict.get, reverse=True)
            # with open(os.path.join(self.ckp_path, 'train_names_sorted_by_dist.json'), 'w') as out_file:
            #     json.dump(sorted_dist_dict, out_file)

    def test_for_select_hip(self):
        self.model.eval()
        with torch.no_grad():
            [h, w] = self.opts['img_size']
            max_dist_dict = dict()
            name_img_dict = dict()
            for step, data in enumerate(tqdm(self.train_loader)):
                img = data['gcn_img'].to(self.device)
                target_control = data['target_control'][0].to(self.device)
                init_control = self.sample_data['gt_ctr'].repeat(img.shape[0], 1, 1)
                gcn_component = {'adj_matrix': data['gcn_component'][0]['adj_matrix'].to(self.device)}
                output = self.model(img, [init_control], [gcn_component])
                pred_control = output['pred_coor'][0][-1]

                for k in range(img.shape[0]):
                    # Compute Hausdorff distance
                    pred = pred_control[k].data.cpu().numpy() * [w, h]
                    target = target_control[k].data.cpu().numpy() * [w, h]
                    init = init_control[k].data.cpu().numpy() * [w, h]

                    # Save prediction json
                    file_name = data['file_name'][k]
                    avg_dist, max_dist = metrics.hausdorff_dist(pred, target)
                    IOU = metrics.get_ctr_iou(np.round(pred).astype(np.int32), np.round(target).astype(np.int32), h, w)
                    max_dist_dict[file_name] = max_dist

                    img = (data['ori_img'][k, ...].cpu().numpy()).astype(np.uint8)
                    img = np.transpose(img, [1, 2, 0])
                    PIL_img = Image.fromarray(img)
                    poly_img = PIL_img.copy()
                    draw_gt = ImageDraw.Draw(poly_img)

                    draw_gt.point([tuple(x) for x in target.astype(np.int32)], fill="green")
                    draw_gt.point([tuple(x) for x in init.astype(np.int32)], fill="red")
                    draw_gt.point([tuple(x) for x in pred.astype(np.int32)], fill="yellow")
                    draw_gt.text((0, 0), file_name, (0, 255, 0), font=ImageFont.load_default())
                    draw_gt.text((0, 12), "IOU: {0:.3f}".format(IOU), (0, 255, 0), font=ImageFont.load_default())
                    draw_gt.text((0, 24), "avg_dist: {0:.3f}".format(avg_dist), (0, 255, 0), font=ImageFont.load_default())
                    draw_gt.text((0, 36), "max_dist: {0:.3f}".format(max_dist), (0, 255, 0), font=ImageFont.load_default())
                    name_img_dict[file_name] = poly_img

            sorted_name_list = sorted(max_dist_dict, key=max_dist_dict.get, reverse=False)
            with open(os.path.join(self.ckp_path, 'names_sorted_by_dist.json'), 'w') as out_file:
                json.dump(sorted_name_list, out_file)
            for i, img_name in enumerate(sorted_name_list):
                print(img_name, max_dist_dict[img_name])
                name_img_dict[img_name].show()
                pass

    def test_raw_img(self):
        print('Testing')
        self.model.eval()
        with torch.no_grad():
            [h, w] = self.opts['img_size']
            for step, data in enumerate(tqdm(self.test_loader)):
                img = data['gcn_img'].to(self.device)
                init_control = self.sample_data['gt_ctr'].repeat(img.shape[0], 1, 1)
                gcn_component = {'adj_matrix': data['gcn_component'][0]['adj_matrix'].to(self.device)}
                output = self.model(img, [init_control], [gcn_component])
                pred_control = output['pred_coor'][0][-1]
                for k in range(img.shape[0]):
                    # Compute Hausdorff distance
                    pred = pred_control[k].data.cpu().numpy() * [w, h]
                    file_name = data['file_name'][k]
                    with open(os.path.join(self.opts['test_write_dir'], file_name, file_name +'_pred_' + os.path.basename(self.ckp_path) + '.json'), 'w') as out_file:
                        json.dump({'pred_control': pred.tolist()}, out_file)


if __name__ == '__main__':
    import torch.backends.cudnn as cudnn
    # cudnn related setting
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    args = get_args()
    args.exp = 'Experiments/hip-contour.json'
    args.resume = 'Checkpoints/hip_06_04_02_23_13/best.pth'
    evaluator = Eval(args)
    # evaluator.validate()
    evaluator.test()
    # evaluator.generate_pred_for_train()
    # evaluator.tmp_test()
    # evaluator.test_raw_img()
