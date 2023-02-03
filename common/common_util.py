from torch.nn import functional as F
import torch
import matplotlib.pyplot as plt

from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import torch.nn as nn
import scipy.stats as st


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    size = nms_radius * 2 + 1
    avg_size = 2
    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=size, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    # max_map = max_pool(scores)

    max_mask = scores == max_pool(scores)
    max_mask_ = torch.rand(max_mask.shape).to(max_mask.device) / 10
    max_mask_[~max_mask] = 0
    mask = ((max_mask_ == max_pool(max_mask_)) & (max_mask_ > 0))

    return torch.where(mask, scores, zeros)


def pre_processing(data):
    """ Enhance retinal images """
    train_imgs = datasets_normalized(data)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)

    train_imgs = train_imgs / 255.

    return train_imgs.astype(np.float32)


def rgb2gray(rgb):
    """ Convert RGB image to gray image """
    r, g, b = rgb.split()
    return g


def clahe_equalized(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    images_equalized = np.empty(images.shape)
    images_equalized[:, :] = clahe.apply(np.array(images[:, :],
                                                  dtype=np.uint8))

    return images_equalized


def datasets_normalized(images):
    # images_normalized = np.empty(images.shape)
    images_std = np.std(images)
    images_mean = np.mean(images)
    images_normalized = (images - images_mean) / (images_std + 1e-6)
    minv = np.min(images_normalized)
    images_normalized = ((images_normalized - minv) /
                         (np.max(images_normalized) - minv)) * 255

    return images_normalized


def adjust_gamma(images, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    new_images = np.empty(images.shape)
    new_images[:, :] = cv2.LUT(np.array(images[:, :],
                                        dtype=np.uint8), table)

    return new_images


def nms(detector_pred, nms_thresh=0.1, nms_size=10, detector_label=None, mask=False):
    """ Apply NMS on predictions, if mask, then remove geo_points that appearing in labels """
    detector_pred = detector_pred.clone().detach()

    B, _, h, w = detector_pred.shape

    # if mask:
    #     assert detector_label is not None
    #     detector_pred[detector_pred < nms_thresh] = 0
    #     label_mask = detector_label
    #
    #     # more area
    #
    #     detector_label = detector_label.long().cpu().numpy()
    #     detector_label = detector_label.astype(np.uint8)
    #     kernel = np.ones((3, 3), np.uint8)
    #     label_mask = np.array([cv2.dilate(detector_label[s, 0], kernel, iterations=1)
    #                            for s in range(len(detector_label))])
    #     label_mask = torch.from_numpy(label_mask).unsqueeze(1)
    #     detector_pred[label_mask > 1e-6] = 0

    scores = simple_nms(detector_pred, nms_size)

    scores = scores.reshape(B, h, w)

    points = [
        torch.nonzero(s > nms_thresh)
        for s in scores]

    scores = [s[tuple(k.t())] for s, k in zip(scores, points)]

    points, scores = list(zip(*[
        remove_borders(k, s, 8, h, w)
        for k, s in zip(points, scores)]))
    points = [torch.flip(k, [1]).long() for k in points]

    return points


def sample_keypoint_desc(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints.clone().float()

    keypoints /= torch.tensor([(w * s - 1), (h * s - 1)]).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)

    args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)

    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


def sample_descriptors(detector_pred, descriptor_pred, affine_descriptor_pred, grid_inverse,
                       nms_size=10, nms_thresh=0.1, scale=8, affine_detector_pred=None):
    """
    sample descriptors based on keypoints
    :param affine_descriptor_pred:
    :param descriptor_pred:
    :param detector_pred:
    :param grid_inverse: used for inverse transformation of affine
    :param nms_size
    :param nms_thresh
    :param scale: down sampling size of detector
    :return: sampled descriptors
    """
    B, _, h, w = detector_pred.shape
    keypoints = nms(detector_pred, nms_size=nms_size, nms_thresh=nms_thresh)

    affine_keypoints = [(grid_inverse[s, k[:, 1].long(), k[:, 0].long()]) for s, k in
                        enumerate(keypoints)]

    kp = []
    affine_kp = []
    for s, k in enumerate(affine_keypoints):
        idx = (k[:, 0] < 1) & (k[:, 0] > -1) & (k[:, 1] < 1) & (
                k[:, 1] > -1)
        kp.append(keypoints[s][idx])
        ak = k[idx]
        ak[:, 0] = (ak[:, 0] + 1) / 2 * (w - 1)
        ak[:, 1] = (ak[:, 1] + 1) / 2 * (h - 1)
        affine_kp.append(ak)

    descriptors = [sample_keypoint_desc(k[None], d[None], s=scale)[0]
                   for k, d in zip(kp, descriptor_pred)]
    affine_descriptors = [sample_keypoint_desc(k[None], d[None], s=scale)[0]
                          for k, d in zip(affine_kp, affine_descriptor_pred)]
    return descriptors, affine_descriptors, keypoints
