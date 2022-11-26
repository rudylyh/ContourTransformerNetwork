import glob
import os
import shutil
import numpy as np
import json
from PIL import Image, ImageDraw

output_dir = '/storage/Experiments/Hand_GCN/train_val_pelvic_512_my/'

np.random.seed(110)
target_size = 512

all_files = glob.glob('/media/smiao/XrayData/Incoming/PelvicXrays/images/*')
file_names1 = '/media/smiao/XrayData/Incoming/PelvicXrays/pelvic_landmark_train.txt'
# file_names2 = '/media/smiao/XrayData/Incoming/PelvicXrays/pelvic_landmark_val.txt'

root_folder = '/media/smiao/XrayData/Incoming/PelvicXrays/images/'
root_folder_anno = '/media/smiao/XrayData/Incoming/PelvicXrays/images_all.anno/'
target_folder = 'pelvis_train_val'

with open(file_names1, 'r') as f:
    file_names1 = f.read().splitlines()

# with open(file_names2, 'r') as f:
#     file_names2 = f.read().splitlines()

# file_names = file_names1 + file_names2
file_names = file_names1
new_file_names = []

for each in file_names:
    new_file = each.split('/')[-1]
    new_file_names.append(new_file)

all_train_files = sorted(np.random.choice(new_file_names, int(len(new_file_names) * 0.8), replace=False).tolist())
all_val_files = sorted([item for item in new_file_names if item not in all_train_files])

for each in new_file_names:
    if each in all_train_files:
        target_dir = output_dir + 'train_val_pelvis/train/' + each.split('.')[0]
    else:
        target_dir = output_dir + 'train_val_pelvis/val/' + each.split('.')[0]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if os.path.exists(root_folder + each):
        shutil.copy(root_folder + each, target_dir)
    if os.path.exists(root_folder_anno + each + '.anno'):
        shutil.copy(root_folder_anno + each + '.anno', target_dir)

all_train_files = glob.glob(output_dir + '/train_val_pelvis/train/*')
all_val_files = glob.glob(output_dir + '/train_val_pelvis/val/*')


initial_controls = {}
mean_controls = {}
file_dicts = {}

# Compute mean shape
for i, each in enumerate(all_train_files):
    print(i, each)
    new_landmarks = {}

    file_name = each.split('/')[-1]
    json_file = each + '/' + file_name + '.png.anno'
    target_root = output_dir + 'train_val_pelvis_{0}/train/'.format(target_size)

    img_file = each + '/' + file_name + '.png'
    img = Image.open(img_file)
    ori_x, ori_y = img.size

    img.thumbnail((target_size, target_size), Image.ANTIALIAS)
    new_x, new_y = img.size

    x_ratio = float(ori_y * target_size / new_y)
    y_ratio = float(ori_x * target_size / new_x)

    with open(json_file, 'r') as f:
        points = json.load(f)

    new_landmarks['target_control'] = []
    new_landmarks['init_control'] = []
    annos = points['annotations']

    for marks in annos:
        new_x = marks['loc'][0] / ori_y
        new_y = marks['loc'][1] / ori_x
        new_landmarks['target_control'].append([new_x, new_y])

        try:
            initial_controls[marks['name']].append([new_x, new_y])
        except:
            initial_controls[marks['name']] = [[new_x, new_y]]

    file_dicts[file_name] = new_landmarks

for key, vals in initial_controls.items():
    vals = np.asarray(vals)
    mean_controls[key] = np.mean(vals, axis=0).tolist()

# Prepare training json files
for i, each in enumerate(all_train_files):
    print(i, each)
    new_landmarks = {}

    file_name = each.split('/')[-1]
    json_file = each + '/' + file_name + '.png.anno'
    target_root = output_dir + '/train_val_pelvis_{0}/train/'.format(target_size)

    img_file = each + '/' + file_name + '.png'
    img = Image.open(img_file)
    ori_W, ori_H = img.size

    img.thumbnail((target_size, target_size), Image.ANTIALIAS)
    new_W, new_H = img.size

    with open(json_file, 'r') as f:
        points = json.load(f)

    new_landmarks['target_control'] = []
    new_landmarks['init_control'] = []
    annos = points['annotations']

    for marks in annos:
        new_x = marks['loc'][0] / ori_H * new_H / target_size
        new_y = marks['loc'][1] / ori_W * new_W / target_size
        new_landmarks['target_control'].append([new_x, new_y])

    mean_control = np.asarray(list(mean_controls.values()))
    mean_control[:, 0] = mean_control[:, 0] * new_H / target_size
    mean_control[:, 1] = mean_control[:, 1] * new_W / target_size

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
    curr_gt = np.asarray(new_landmarks['target_control']) * target_size
    draw_gt.point([tuple(x) for x in curr_gt.astype(np.int32)], fill="red")
    draw_gt.point([tuple(x) for x in curr_init.astype(np.int32)], fill="green")
    poly_img.save(target_dir + '/' + file_name + '_visual.png')

    with open(json_target, 'w') as f:
        json.dump(new_landmarks, f, indent=4)

# Prepare validation json
for i, each in enumerate(all_val_files):
    print(i, each)
    new_landmarks = {}

    file_name = each.split('/')[-1]
    json_file = each + '/' + file_name + '.png.anno'
    target_root = output_dir + './train_val_pelvis_{0}/val/'.format(target_size)

    img_file = each + '/' + file_name + '.png'
    img = Image.open(img_file)
    ori_W, ori_H = img.size

    img.thumbnail((target_size, target_size), Image.ANTIALIAS)
    new_W, new_H = img.size

    with open(json_file, 'r') as f:
        points = json.load(f)

    new_landmarks['target_control'] = []
    new_landmarks['init_control'] = []
    annos = points['annotations']

    for marks in annos:
        new_x = marks['loc'][0] / ori_H * new_H / target_size
        new_y = marks['loc'][1] / ori_W * new_W / target_size
        new_landmarks['target_control'].append([new_x, new_y])

    mean_control = np.asarray(list(mean_controls.values()))
    mean_control[:, 0] = mean_control[:, 0] * new_H / target_size
    mean_control[:, 1] = mean_control[:, 1] * new_W / target_size

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
    curr_gt = np.asarray(new_landmarks['target_control']) * target_size
    draw_gt.point([tuple(x) for x in curr_gt.astype(np.int32)], fill="red")
    draw_gt.point([tuple(x) for x in curr_init.astype(np.int32)], fill="green")
    poly_img.save(target_dir + '/' + file_name + '_visual.png')

    with open(json_target, 'w') as f:
        json.dump(new_landmarks, f, indent=4)
