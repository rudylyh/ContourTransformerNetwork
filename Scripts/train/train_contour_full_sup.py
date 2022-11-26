import torch
import json
import os
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
import time
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
from DataProvider import knuckle_contour_data_full_sup
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, args):
        self.opts = json.load(open(args.exp, 'r'))
        self.global_step = 0
        self.best_performance = 1000.0 # max_dist
        self.current_performance = 1000.0
        self.device = torch.device("cuda:{}".format(self.opts['gpus'][0]))

        ### Save checkpoint
        self.save_dir = self.opts['exp_dir'] + time.strftime("%m_%d_%H_%M_%S")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        os.system('cp %s %s' % (args.exp, self.save_dir))
        os.system('cp %s %s' % ('./Evaluation/losses.py', self.save_dir))
        os.system('cp %s %s' % ('./Scripts/train/train_contour.py', self.save_dir))
        os.system('cp %s %s' % ('./DataProvider/'+self.opts['data']+'_contour_data.py', self.save_dir))

        ### Prepare dataset
        self.train_writer = SummaryWriter(os.path.join(self.save_dir, 'logs', 'train'))
        self.val_writer = SummaryWriter(os.path.join(self.save_dir, 'logs', 'val'))

        train_data = knuckle_contour_data_full_sup.SelfDataset(mode='train', opts=self.opts)
        self.train_loader = DataLoader(train_data, batch_size=self.opts['dataset']['train']['batch_size'],
                            shuffle=True,
                            num_workers=self.opts['dataset']['train']['num_workers'],
                            pin_memory=True)
        val_data = knuckle_contour_data_full_sup.SelfDataset(mode='val', opts=self.opts)
        self.val_loader = DataLoader(val_data, batch_size=self.opts['dataset']['val']['batch_size'],
                            shuffle=False,
                            num_workers=self.opts['dataset']['val']['num_workers'],
                            pin_memory=True)

        ### Initialize model
        self.model = model_gnn.PolyGNN(gcn_out_dim=self.opts['gcn_out_dim'], gcn_steps=self.opts['gcn_steps'], nInputChannels=self.opts['nInputChannels'], opts=self.opts)
        self.model.encoder.reload(self.opts['encoder_reload'])
        self.optimizer = trainUtil.init_optimizer(self.opts['optimizer'], self.opts['lr'], self.opts['weight_decay'], self.model)
        # self.lr_decay = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opts['lr_decay'], gamma=self.opts['gamma'])
        self.model = convert_model(self.model)
        self.model = torch.nn.DataParallel(self.model.to(self.device), device_ids=self.opts['gpus'])
        self.vgg_model = Vgg16().type(torch.cuda.FloatTensor).to(self.device)
        if args.resume is not None:
            self.resume(args.resume)


    def save_checkpoint(self, epoch):
        save_state = {
            'best_performance': self.best_performance,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(save_state, os.path.join(self.save_dir, 'best.pth'))
        print('Saved model best.pth \n')


    def resume(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict['state_dict'])
        self.best_performance = state_dict['best_performance']


    def train(self, epoch):
        self.model.train()
        for step, data in enumerate(self.train_loader):
            ### Prepare inputs
            img = data['gcn_img'].to(self.device)
            grad_mag = data['grad_mag'].to(self.device)
            target_control = data['target_control'][0].to(self.device)
            init_control = data['init_control'][0].to(self.device)
            gcn_component = {'adj_matrix': data['gcn_component'][0]['adj_matrix'].to(self.device)}

            ### Forward
            self.optimizer.zero_grad()
            output = self.model(img, [init_control], [gcn_component])
            pred_control = output['pred_coor'][0][-1]

            ### Losses and backward
            final_loss = torch.mean(losses.poly_matching_loss_orderedinput(pred_control, target_control, loss_type=self.opts['loss_type']))
            final_loss.backward()
            self.optimizer.step()

            ### Logs
            self.train_writer.add_scalar('final_loss', final_loss, self.global_step)
            if self.global_step % self.opts['print_freq'] == 0:
                print("{0} Epoch:{1}, Step:{2}, LR:{3}, Loss:{4:.4f}".
                      format(str(datetime.now()), epoch, self.global_step, self.optimizer.param_groups[0]['lr'], final_loss))

            ### Validation
            if self.global_step % self.opts['val_freq'] == 0:
                self.model.eval()
                self.validate()
                self.model.train()
                if self.current_performance < self.best_performance:
                    self.best_performance = self.current_performance
                    self.save_checkpoint(epoch)

            self.global_step += 1


    def validate(self):
        with torch.no_grad():
            val_loss, val_avg_dist, val_max_dist, val_IOU, num_imgs = 0.0, 0.0, 0.0, 0.0, 0
            val_perceptual_loss, val_shape_loss, val_grad_loss = 0.0, 0.0, 0.0
            [res_rows, res_cols] = self.opts['val_img_num']
            [h, w] = self.opts['img_size']
            pred_result_img = Image.new('RGB', (res_cols * w, res_rows * h))

            for step, data in enumerate(self.val_loader):
                img = data['gcn_img'].to(self.device)
                grad_mag = data['grad_mag'].to(self.device)
                target_control = data['target_control'][0].to(self.device)
                init_control = data['init_control'][0].to(self.device)
                gcn_component = {'adj_matrix': data['gcn_component'][0]['adj_matrix'].to(self.device)}

                output = self.model(img, [init_control], [gcn_component])
                num_imgs += img.shape[0]
                pred_control = output['pred_coor'][0][-1]
                # pred_control = target_control

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
                    # val_perceptual_loss += perceptual_loss[k]
                    # val_shape_loss += shape_loss[k]
                    # val_grad_loss += grad_loss[k]
                    # val_loss += final_loss[k]

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
                    draw_gt.text((0, 12), "IoU: {0:.3f}".format(IOU), (0, 255, 0), font=ImageFont.load_default())
                    draw_gt.text((0, 24), "avg_dist: {0:.3f}".format(avg_dist), (0, 255, 0), font=ImageFont.load_default())
                    draw_gt.text((0, 36), "max_dist: {0:.3f}".format(max_dist), (0, 255, 0), font=ImageFont.load_default())

                    img_idx = step * self.opts['dataset']['val']['batch_size'] + k
                    row = img_idx // res_cols
                    col = img_idx % res_cols
                    pred_result_img.paste(poly_img, (col * w, row * h))

            self.current_performance = val_max_dist/num_imgs
            print('\rValidate on val set: best_max_dist: {0:.4f}, avg_dist: {1:.4f}, max_dist: {2:.4f}, IOU: {3:.4f}\n'
                .format(self.best_performance, val_avg_dist/num_imgs, val_max_dist/num_imgs, val_IOU/num_imgs))

            pred_result_img.save(os.path.join(self.save_dir, 'val_step{0}.png'.format(self.global_step)))
            pred_result_img = np.array(pred_result_img).transpose((2, 0, 1))

            self.val_writer.add_scalar('val_avg_dist', val_avg_dist/num_imgs, self.global_step)
            self.val_writer.add_scalar('val_max_dist', val_max_dist/num_imgs, self.global_step)
            self.val_writer.add_scalar('val_IOU', val_IOU/num_imgs, self.global_step)


if __name__ == '__main__':
    import torch.backends.cudnn as cudnn
    # cudnn related setting
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    print('==> Parsing Args')
    args = get_args()
    # args.exp = 'Experiments/knuckle-contour.json'
    # args.resume = 'Checkpoints/knuckle_11_06_19_35_18/best.pth'

    print('==> Init Trainer')
    trainer = Trainer(args)

    print('==> Start Loop over trainer\n')
    for epoch in range(0, trainer.opts['max_epochs']):
        trainer.train(epoch)
