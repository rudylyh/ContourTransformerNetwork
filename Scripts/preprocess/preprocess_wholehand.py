import glob
import os
import shutil
import numpy as np
import json
from PIL import Image, ImageDraw

output_dir = '/storage/Experiments/Hand_GCN/Data_Prepared/'

np.random.seed(110)
target_size = 512

all_files = glob.glob('/storage/Rheumatology/PAII-1-130/Hand_PNG/*')
root_folder = '/storage/Rheumatology/PAII-1-130/Hand_PNG/'
new_file_names = set()
all_patient_nums = set()

for each in all_files:
    file_name = each.split('/')[-1].split('.')[0]
    new_file_names.add(file_name)
    all_patient_nums.add(file_name.split('-')[0])

new_file_names = list(new_file_names)
all_patient_nums = list(all_patient_nums)

all_train_files = sorted(np.random.choice(all_patient_nums, int(len(all_patient_nums) * 0.8), replace=False).tolist())
all_val_files = sorted([item for item in all_patient_nums if item not in all_train_files])

for each in new_file_names:
    patient_num = each.split('-')[0]
    if patient_num in all_train_files:
        target_dir = output_dir + 'train_val_wholehand/train/' + each
    else:
        target_dir = output_dir + 'train_val_wholehand/val/' + each

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if os.path.exists(root_folder + each + '.png'):
        shutil.copy(root_folder + each + '.png', target_dir)
        shutil.copy(root_folder + each + '.json', target_dir)


################# Compute Meanshape ###########################
all_train_files = glob.glob(output_dir + '/train_val_wholehand/train/*')
all_val_files = glob.glob(output_dir + '/train_val_wholehand/val/*')
initial_controls_L = {}
initial_controls_R = {}
mean_controls_L = {}
mean_controls_R = {}
file_dicts = {}

# Compute mean shape
for i, each in enumerate(all_train_files):
    print(i, each)
    new_landmarks = {}

    file_name = each.split('/')[-1]
    left_or_right = file_name.split('-')[1]
    json_file = each + '/' + file_name + '.json'
    target_root = output_dir + '/train_val_wholehand_{0}/train/'.format(target_size)

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
    fingers = points['shapes']

    counter = 0
    for k, finger in enumerate(fingers):
        landmarks = finger['points']
        name = finger['label']

        for j, each in enumerate(landmarks):
            new_x = each[0] / ori_y
            new_y = each[1] / ori_x
            new_landmarks['target_control'].append([new_x, new_y])

            if left_or_right == 'HL':
                try:
                    initial_controls_L[counter].append([new_x, new_y])
                except:
                    initial_controls_L[counter] = [[new_x, new_y]]
            else:
                try:
                    initial_controls_R[counter].append([new_x, new_y])
                except:
                    initial_controls_R[counter] = [[new_x, new_y]]
            counter += 1

    file_dicts[file_name] = new_landmarks

for key, vals in initial_controls_L.items():
    vals = np.asarray(vals)
    mean_controls_L[key] = np.mean(vals, axis=0).tolist()

for key, vals in initial_controls_R.items():
    vals = np.asarray(vals)
    mean_controls_R[key] = np.mean(vals, axis=0).tolist()

################# Compute and store init control points and target control points
for i, each in enumerate(all_train_files):
    print(i, each)
    new_landmarks = {}

    file_name = each.split('/')[-1]
    left_or_right = file_name.split('-')[1]
    json_file = each + '/' + file_name + '.json'
    target_root = output_dir + '/train_val_wholehand_{0}/train/'.format(target_size)

    img_file = each + '/' + file_name + '.png'
    img = Image.open(img_file)
    ori_W, ori_H = img.size

    img.thumbnail((target_size, target_size), Image.ANTIALIAS)
    new_W, new_H = img.size

    with open(json_file, 'r') as f:
        points = json.load(f)

    new_landmarks['target_control'] = []
    new_landmarks['init_control'] = []

    fingers = points['shapes']

    for k, finger in enumerate(fingers):
        landmarks = finger['points']
        name = finger['label']

        for j, each in enumerate(landmarks):
            new_x = each[0] / ori_H * new_H / target_size
            new_y = each[1] / ori_W * new_W / target_size
            new_landmarks['target_control'].append([new_x, new_y])

            if left_or_right == 'HL':
                mean_control = np.asarray(list(mean_controls_L.values()))
                mean_control[:, 0] = mean_control[:, 0] * new_H / target_size
                mean_control[:, 1] = mean_control[:, 1] * new_W / target_size
                new_landmarks['init_control'] = mean_control.tolist()
            else:
                mean_control = np.asarray(list(mean_controls_R.values()))
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

    for w, x in enumerate(curr_gt.astype(np.int32)):
        draw_gt.point([tuple(x)], fill=(255, w * 8, w * 8))
    draw_gt.point([tuple(x) for x in curr_init.astype(np.int32)], fill="green")
    poly_img.save(target_dir + '/' + file_name + '_visual.png')

    with open(json_target, 'w') as f:
        json.dump(new_landmarks, f, indent=4)

# Prepare validation json
for i, each in enumerate(all_val_files):
    print(i, each)
    new_landmarks = {}

    file_name = each.split('/')[-1]
    left_or_right = file_name.split('-')[1]
    json_file = each + '/' + file_name + '.json'
    target_root = output_dir + '/train_val_wholehand_{0}/val/'.format(target_size)

    img_file = each + '/' + file_name + '.png'
    img = Image.open(img_file)
    ori_W, ori_H = img.size

    img.thumbnail((target_size, target_size), Image.ANTIALIAS)
    new_W, new_H = img.size

    with open(json_file, 'r') as f:
        points = json.load(f)

    new_landmarks['target_control'] = []
    new_landmarks['init_control'] = []

    fingers = points['shapes']

    for k, finger in enumerate(fingers):
        landmarks = finger['points']
        name = finger['label']

        for j, each in enumerate(landmarks):
            new_x = each[0] / ori_H * new_H / target_size
            new_y = each[1] / ori_W * new_W / target_size
            new_landmarks['target_control'].append([new_x, new_y])

            if left_or_right == 'HL':
                mean_control = np.asarray(list(mean_controls_L.values()))
                mean_control[:, 0] = mean_control[:, 0] * new_H / target_size
                mean_control[:, 1] = mean_control[:, 1] * new_W / target_size
                new_landmarks['init_control'] = list(mean_controls_L.values())
            else:
                mean_control = np.asarray(list(mean_controls_R.values()))
                mean_control[:, 0] = mean_control[:, 0] * new_H / target_size
                mean_control[:, 1] = mean_control[:, 1] * new_W / target_size
                new_landmarks['init_control'] = list(mean_controls_R.values())

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


