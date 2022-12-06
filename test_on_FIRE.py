import configparser

import numpy as np
from tqdm import tqdm

from common.eval_util import compute_auc
from predictor import Predictor
import os
import cv2
import yaml

config_path = './config/test.yaml'
if os.path.exists(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
else:
    raise FileNotFoundError("Config File doesn't Exist")

Pred = Predictor(config)

data_path = './data/'  # Change the data_path according to your own setup
testset = 'FIRE'
use_matching_trick = config['PREDICT']['use_matching_trick']
gt_dir = os.path.join(data_path, testset, 'Ground Truth')
im_dir = os.path.join(data_path, testset, 'Images')

match_pairs = [x for x in os.listdir(gt_dir) if x.endswith('.txt')
               and not x.endswith('P37_1_2.txt')]

match_pairs.sort()
big_num = 1e6
good_nums_rate = []
image_num = 0

failed = 0
inaccurate = 0
mae = 0
mee = 0

# category: S, P, A, corresponding to Easy, Hard, Mod in paper
auc_record = dict([(category, []) for category in ['S', 'P', 'A']])

for pair_file in tqdm(match_pairs):
    gt_file = os.path.join(gt_dir, pair_file)
    file_name = pair_file.replace('.txt', '')

    category = file_name.split('_')[2][0]

    refer = file_name.split('_')[2] + '_' + file_name.split('_')[3]
    query = file_name.split('_')[2] + '_' + file_name.split('_')[4]

    query_im_path = os.path.join(im_dir, query + '.jpg')
    refer_im_path = os.path.join(im_dir, refer + '.jpg')
    H_m1, inliers_num_rate, query_image, _ = Pred.compute_homography(query_im_path, refer_im_path)
    H_m2 = None
    if use_matching_trick:
        if H_m1 is not None:
            h, w = Pred.image_height, Pred.image_width
            query_align_first = cv2.warpPerspective(query_image, H_m1, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=(0))
            query_align_first = query_align_first.astype(float)
            query_align_first /= 255.
            H_m2, inliers_num_rate, _, _ = Pred.compute_homography(query_align_first, refer_im_path, query_is_image=True)

    good_nums_rate.append(inliers_num_rate)
    image_num += 1

    if inliers_num_rate < 1e-6:
        failed += 1
        avg_dist = big_num
    else:
        points_gd = np.loadtxt(gt_file)
        raw = np.zeros([len(points_gd), 2])
        dst = np.zeros([len(points_gd), 2])
        raw[:, 0] = points_gd[:, 2]
        raw[:, 1] = points_gd[:, 3]
        dst[:, 0] = points_gd[:, 0]
        dst[:, 1] = points_gd[:, 1]
        dst_pred = cv2.perspectiveTransform(raw.reshape(-1, 1, 2), H_m1)
        if H_m2 is not None:
            dst_pred = cv2.perspectiveTransform(dst_pred.reshape(-1, 1, 2), H_m2)

        dst_pred = dst_pred.squeeze()

        dis = (dst - dst_pred) ** 2
        dis = np.sqrt(dis[:, 0] + dis[:, 1])
        avg_dist = dis.mean()
        
        mae = dis.max()
        mee = np.median(dis)
        if mae > 50 or mee > 20:
            inaccurate += 1
    
    auc_record[category].append(avg_dist)
    

print('-'*40)
print(f"Failed:{'%.2f' % (100*failed/image_num)}%, Inaccurate:{'%.2f' % (100*inaccurate/image_num)}%, "
      f"Acceptable:{'%.2f' % (100*(image_num-inaccurate-failed)/image_num)}%")

print('-'*40)

auc = compute_auc(auc_record['S'], auc_record['P'], auc_record['A'])
print('S: %.3f, P: %.3f, A: %.3f, mAUC: %.3f' % (auc['s'], auc['p'], auc['a'], auc['mAUC']))
