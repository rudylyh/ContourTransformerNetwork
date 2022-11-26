import glob
import os
import shutil
import numpy as np
import json
from PIL import Image, ImageDraw

experiment_json = '../train_val_aasce_512/val/sunhl-1th-01-Mar-2017-310 a ap/' \
                  'sunhl-1th-01-Mar-2017-310 a ap.json'

all_folders = glob.glob('../train_val_aasce_512/val/*')

for each in all_folders:
    filename = each.split('/')[-1]
    experiment_json = each + '/' + filename + '.json'

    with open(experiment_json, 'r') as f:
        points = json.load(f)
        init_control = np.asarray(points['init_control'])
        init_control[:, 0] = 0

    with open(experiment_json, 'w') as f:
        points['init_control'] = init_control.tolist()
        json.dump(points, f)
