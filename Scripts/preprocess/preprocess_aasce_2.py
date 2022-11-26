import glob
import os
import shutil
import numpy as np
import json
from PIL import Image, ImageDraw

np.random.seed(110)

all_train_files = glob.glob('../train_val_aasce/train/*')
all_val_files = glob.glob('../train_val_aasce/val/*')
all_test_files = glob.glob('../train_val_aasce/test/*')

target_size = 512
initial_controls = {}
mean_controls = {}
file_dicts = {}


def scale_init(init_control, scale):
    o = (init_control.min(axis=0) + init_control.max(axis=0)) * .5
    init_control = o * (1 - scale) + init_control * scale
    return init_control


# for i, each in enumerate(all_train_files):
#     print(i, each)
#     new_landmarks = {}
#
#     file_name = each.split('/')[-1]
#     json_file = each + '/' + file_name + '.json'
#     target_root = '../train_val_aasce_{0}/train/'.format(target_size)
#
#     img_file = each + '/' + file_name + '.jpg'
#     img = Image.open(img_file)
#     W, H = img.size
#
#     img.thumbnail((target_size, target_size), Image.ANTIALIAS)
#     new_W, new_H = img.size
#
#     w_ratio = float(W / new_W)
#     h_ratio = float(H / new_H)
#
#     with open(json_file, 'r') as f:
#         points = json.load(f)
#
#     new_landmarks['target_control'] = []
#     annos = points['target_control']
#
#     for m, marks in enumerate(annos):
#         new_x = marks[0]
#         new_y = marks[1]
#         new_landmarks['target_control'].append([new_x, new_y])
#
#         try:
#             initial_controls[m].append([new_x, new_y])
#         except:
#             initial_controls[m] = [[new_x, new_y]]
#
#     file_dicts[file_name] = new_landmarks
#
# for key, vals in initial_controls.items():
#     vals = np.asarray(vals)
#     mean_controls[key] = np.mean(vals, axis=0).tolist()
#
# with open('aasce_{0}_mean_control.json'.format(target_size), 'w') as f:
#     json.dump(mean_controls, f, indent=4)


with open('aasce_{0}_mean_control.json'.format(target_size), 'r') as f:
    mean_controls = json.load(f)

# for i, each in enumerate(all_train_files):
#     print(i, each)
#     new_landmarks = {}
#
#     file_name = each.split('/')[-1]
#     json_file = each + '/' + file_name + '.json'
#     target_root = '../train_val_aasce_{0}/train/'.format(target_size)
#
#     img_file = each + '/' + file_name + '.jpg'
#     img = Image.open(img_file)
#     W, H = img.size
#
#     img.thumbnail((target_size, target_size), Image.ANTIALIAS)
#     new_W, new_H = img.size
#
#     w_ratio = float(W / new_W)
#     h_ratio = float(H / new_H)
#
#     with open(json_file, 'r') as f:
#         points = json.load(f)
#
#     new_landmarks['target_control'] = []
#     new_landmarks['init_control'] = []
#     annos = points['target_control']
#
#     for marks in annos:
#         new_x = marks[0] * new_W / target_size
#         new_y = marks[1] * new_H / target_size
#         new_landmarks['target_control'].append([new_x, new_y])
#
#     mean_control = np.asarray(list(mean_controls.values()))
#     mean_control[:, 0] = mean_control[:, 0] * new_W / target_size
#     mean_control[:, 1] = mean_control[:, 1] * new_H / target_size
#
#     new_landmarks['init_control'] = mean_control.tolist()
#
#     a = np.asarray(new_landmarks['init_control'])
#     b = np.asarray(new_landmarks['target_control'])
#
#     img_resize = Image.new("RGB", (target_size, target_size))
#     img_resize.paste(img, (0, 0))
#
#     target_dir = target_root + file_name
#
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#
#     img_resize.save(target_dir + '/' + file_name + '.png')
#     json_target = target_dir + '/' + file_name + '.json'
#
#     poly_img = img_resize.copy()
#     draw_gt = ImageDraw.Draw(poly_img)
#     curr_init = np.asarray(new_landmarks['init_control']) * target_size
#     curr_gt = np.asarray(new_landmarks['target_control']) * target_size
#
#     draw_gt.point([tuple(x) for x in curr_gt.astype(np.int32)], fill="red")
#     draw_gt.point([tuple(x) for x in curr_init.astype(np.int32)], fill="green")
#     poly_img.save(target_dir + '/' + file_name + '_visual.png')
#
#     with open(json_target, 'w') as f:
#         json.dump(new_landmarks, f, indent=4)

# for i, each in enumerate(all_val_files):
#     print(i, each)
#     new_landmarks = {}
#
#     file_name = each.split('/')[-1]
#     json_file = each + '/' + file_name + '.json'
#     target_root = '../train_val_aasce_{0}/val/'.format(target_size)
#
#     img_file = each + '/' + file_name + '.jpg'
#     img = Image.open(img_file)
#     W, H = img.size
#
#     img.thumbnail((target_size, target_size), Image.ANTIALIAS)
#     new_W, new_H = img.size
#
#     w_ratio = float(W / new_W)
#     h_ratio = float(H / new_H)
#
#     with open(json_file, 'r') as f:
#         points = json.load(f)
#
#     new_landmarks['target_control'] = []
#     new_landmarks['init_control'] = []
#     annos = points['target_control']
#
#     for marks in annos:
#         new_x = marks[0] * new_W / target_size
#         new_y = marks[1] * new_H / target_size
#         new_landmarks['target_control'].append([new_x, new_y])
#
#     mean_control = np.asarray(list(mean_controls.values()))
#     mean_control[:, 0] = mean_control[:, 0] * new_W / target_size
#     mean_control[:, 1] = mean_control[:, 1] * new_H / target_size
#
#     new_landmarks['init_control'] = mean_control.tolist()
#
#     a = np.asarray(new_landmarks['init_control'])
#     b = np.asarray(new_landmarks['target_control'])
#
#     img_resize = Image.new("RGB", (target_size, target_size))
#     img_resize.paste(img, (0, 0))
#
#     target_dir = target_root + file_name
#
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#
#     img_resize.save(target_dir + '/' + file_name + '.png')
#     json_target = target_dir + '/' + file_name + '.json'
#
#     poly_img = img_resize.copy()
#     draw_gt = ImageDraw.Draw(poly_img)
#     curr_init = np.asarray(new_landmarks['init_control']) * target_size
#     curr_gt = np.asarray(new_landmarks['target_control']) * target_size
#
#     draw_gt.point([tuple(x) for x in curr_gt.astype(np.int32)], fill="red")
#     draw_gt.point([tuple(x) for x in curr_init.astype(np.int32)], fill="green")
#     poly_img.save(target_dir + '/' + file_name + '_visual.png')
#
#     with open(json_target, 'w') as f:
#         json.dump(new_landmarks, f, indent=4)


for i, each in enumerate(all_test_files):
    print(i, each)
    new_landmarks = {}

    file_name = each.split('/')[-1]
    json_file = each + '/' + file_name + '.json'
    target_root = '../train_val_aasce_{0}/test/'.format(target_size)

    img_file = each + '/' + file_name + '.jpg'
    img = Image.open(img_file)
    W, H = img.size

    # if file_name != '01-July-2019-56':
    #     continue
    # Remove black regions to the left
    img_array = np.asarray(img)
    if len(img_array.shape) == 3:
        img_array = img_array[:, :, 0]
    blacks = (img_array == 0)
    h_percentage = np.sum(blacks, axis=0) / H
    w = next((a for a, x in enumerate(h_percentage) if x < 0.5), 0)
    w_percentage = np.sum(blacks, axis=1) / W
    h = next((b for b, x in enumerate(w_percentage) if x < 0.5), 0)
    img = img.crop((w, h, W, H))

    W, H = img.size
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    blacks = (img_array == 0)
    h_percentage = np.sum(blacks, axis=0) / H
    w = next((a for a, x in enumerate(h_percentage) if x < 0.5), 0)
    w_percentage = np.sum(blacks, axis=1) / W
    h = next((b for b, x in enumerate(w_percentage) if x < 0.5), 0)
    img = img.crop((w, h, W, H))

    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    W, H = img.size

    left = W * 0.1
    top = H * 0.12
    right = W * 0.83
    bottom = H * 0.8
    img = img.crop((left, top, right, bottom))

    W, H = img.size

    img.thumbnail((target_size, target_size), Image.ANTIALIAS)
    new_W, new_H = img.size

    with open(json_file, 'r') as f:
        points = json.load(f)

    new_landmarks['target_control'] = []
    new_landmarks['init_control'] = []

    mean_control = np.asarray(list(mean_controls.values()))
    mean_control[:, 0] = mean_control[:, 0] * new_W / target_size
    mean_control[:, 1] = mean_control[:, 1] * new_H / target_size

    mean_control = scale_init(mean_control, 1)
    new_landmarks['init_control'] = mean_control.tolist()

    a = np.asarray(new_landmarks['init_control'])
    b = np.asarray(new_landmarks['target_control'])

    img_resize = Image.new("RGB", (target_size, target_size))
    img_resize.paste(img, (0, 0))

    target_dir = target_root + file_name

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    img_resize.save(target_dir + '/' + file_name + '.png')
    json_target = target_dir + '/' + file_name + '.json'

    poly_img = img_resize.copy()
    draw_gt = ImageDraw.Draw(poly_img)
    curr_init = np.asarray(new_landmarks['init_control']) * target_size
    draw_gt.point([tuple(x) for x in curr_init.astype(np.int32)], fill="green")

    poly_img.save(target_dir + '/' + file_name + '_visual.png')

    with open(json_target, 'w') as f:
        json.dump(new_landmarks, f, indent=4)