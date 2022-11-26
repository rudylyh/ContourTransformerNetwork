import glob
import os
import shutil
import numpy as np
import json
from PIL import Image

root_dir = '../boostnet_labeldata'
train_files = root_dir + '/' + 'labels/training/filenames.csv'
val_files = root_dir + '/' + 'labels/test/filenames.csv'
train_landmarks = root_dir + '/' + 'labels/training/landmarks.csv'
val_landmarks = root_dir + '/' + 'labels/test/landmarks.csv'

# with open(train_landmarks, 'r')as f1, open(train_files, 'r') as f2:
#     for landmark, filename in zip(f1, f2):
#         result_dict = {}
#         filename = filename.strip().split('.jpg')[0]
#         img_file = root_dir + '/' + 'data/training/' + filename + '.jpg'
#         ori_img = Image.open(img_file)
#         x, y = ori_img.size
#
#         landmarks = [float(x) for x in landmark.split(',')]
#         landmarks_x = landmarks[:68]
#         landmarks_x = [a for a in landmarks_x]
#         landmarks_y = landmarks[68:]
#         landmarks_y = [a for a in landmarks_y]
#         landmarks = np.asarray(list((zip(landmarks_x, landmarks_y))))
#
#         result_dict['target_control'] = landmarks.tolist()
#
#         target_dir = '../train_val_aasce/train/' + filename
#
#         if not os.path.exists(target_dir):
#             os.makedirs(target_dir)
#
#         if os.path.exists(img_file):
#             shutil.move(img_file, target_dir)
#
#         json_target = target_dir + '/' + filename + '.json'
#
#         with open(json_target, 'w') as f:
#             json.dump(result_dict, f, indent=4)

# with open(val_landmarks, 'r')as f1, open(val_files, 'r') as f2:
#     for landmark, filename in zip(f1, f2):
#         result_dict = {}
#         filename = filename.strip().split('.jpg')[0]
#         img_file = root_dir + '/' + 'data/test/' + filename + '.jpg'
#         ori_img = Image.open(img_file)
#         x, y = ori_img.size
#
#         landmarks = [float(x) for x in landmark.split(',')]
#         landmarks_x = landmarks[:68]
#         landmarks_x = [a for a in landmarks_x]
#         landmarks_y = landmarks[68:]
#         landmarks_y = [a for a in landmarks_y]
#         landmarks = np.asarray(list((zip(landmarks_x, landmarks_y))))
#
#         result_dict['target_control'] = landmarks.tolist()
#
#         target_dir = '../train_val_aasce/val/' + filename
#
#         if not os.path.exists(target_dir):
#             os.makedirs(target_dir)
#
#         if os.path.exists(img_file):
#             shutil.move(img_file, target_dir)
#
#         json_target = target_dir + '/' + filename + '.json'
#
#         with open(json_target, 'w') as f:
#             json.dump(result_dict, f, indent=4)

test_files = glob.glob('../test/*')

for img_file in test_files:
    result_dict = {}
    filename = img_file.split('/')[-1].split('.jpg')[0]
    ori_img = Image.open(img_file)
    x, y = ori_img.size

    result_dict['target_control'] = []

    target_dir = '../train_val_aasce/test/' + filename

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if os.path.exists(img_file):
        shutil.move(img_file, target_dir)

    json_target = target_dir + '/' + filename + '.json'

    with open(json_target, 'w') as f:
        json.dump(result_dict, f, indent=4)