import shutil

import torch
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

import numpy as np
from predictor import Predictor
import os
import yaml
from tqdm import tqdm

config_path = './config/test_VARIA.yaml'
if os.path.exists(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError("Config File doesn't Exist")

Pred = Predictor(config)

data_path = './data/'  # Change the data_path according to your own setup
testset = 'VARIA'
save_tmp_info = 'VARIA_info'  # Save keypoints and descriptors
id_path = os.path.join(data_path, testset, 'pair_index.txt')
im_dir = os.path.join(data_path, testset, 'Images')

save_tmp_info = os.path.join(data_path, testset, save_tmp_info)

if os.path.exists(save_tmp_info):
    shutil.rmtree(save_tmp_info)
os.makedirs(save_tmp_info)

with open(id_path, 'r') as f:
    pairs = f.readlines()

inliers = []
classes = []

print('Getting Predictions of All Images')
for img in tqdm(os.listdir(im_dir)):
    if img.endswith('.pgm'):
        image_path = os.path.join(im_dir, img)
        save_path = os.path.join(save_tmp_info, img.replace('.pgm', '.pt'))

        Pred.model_run_one_image(image_path, save_path)

print('Matching Pairs')
for pair in tqdm(pairs):
    pair = pair.strip()
    query, refer, is_accepted = pair.split(', ')

    query_im_path = os.path.join(im_dir, query)
    refer_im_path = os.path.join(im_dir, refer)

    query_info_path = os.path.join(save_tmp_info, query.replace('.pgm', '.pt'))
    refer_info_path = os.path.join(save_tmp_info, refer.replace('.pgm', '.pt'))

    query_info = torch.load(query_info_path)
    refer_info = torch.load(refer_info_path)

    _, inliers_num = Pred.homography_from_tensor(query_info, refer_info)

    inliers.append(inliers_num)
    classes.append(int(is_accepted))

inliers = np.array(inliers)
classes = np.array(classes)

fpr, tpr, threshold = roc_curve(classes, inliers)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, threshold)(eer)

print('VARIA DATASET')
print('EER: %.2f%%, threshold: %d' % (eer*100, thresh))

