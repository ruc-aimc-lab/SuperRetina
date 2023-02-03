import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from common.common_util import sample_keypoint_desc, nms
from model.record_module import update_value_map


def mapping_points(grid, points, h, w):
    """ Using grid_inverse to apply affine transform on geo_points
        :return point set and its corresponding affine point set
    """

    grid_points = [(grid[s, k[:, 1].long(), k[:, 0].long()]) for s, k in
                   enumerate(points)]
    filter_points = []
    affine_points = []
    for s, k in enumerate(grid_points):  # filter bad geo_points
        idx = (k[:, 0] < 1) & (k[:, 0] > -1) & (k[:, 1] < 1) & (
                k[:, 1] > -1)
        gp = grid_points[s][idx]
        gp[:, 0] = (gp[:, 0] + 1) / 2 * (w - 1)
        gp[:, 1] = (gp[:, 1] + 1) / 2 * (h - 1)
        affine_points.append(gp)
        filter_points.append(points[s][idx])

    return filter_points, affine_points


def content_filter(descriptor_pred, affine_descriptor_pred, geo_points,
                   affine_geo_points, content_thresh=0.7, scale=8):
    """
    content-based matching in paper
    :param descriptor_pred: descriptors of input_image images
    :param affine_descriptor_pred: descriptors of affine images
    :param geo_points: 
    :param affine_geo_points:
    :param content_thresh:
    :param scale: down sampling size of descriptor_pred
    :return: content-filtered keypoints
    """

    descriptors = [sample_keypoint_desc(k[None], d[None], scale)[0].permute(1, 0)
                   for k, d in zip(geo_points, descriptor_pred)]
    aff_descriptors = [sample_keypoint_desc(k[None], d[None], scale)[0].permute(1, 0)
                       for k, d in zip(affine_geo_points, affine_descriptor_pred)]
    content_points = []
    affine_content_points = []
    dist = [torch.norm(descriptors[d][:, None] - aff_descriptors[d], dim=2, p=2)
            for d in range(len(descriptors))]
    for i in range(len(dist)):
        D = dist[i]
        if len(D) <= 1:
            content_points.append([])
            affine_content_points.append([])
            continue
        val, ind = torch.topk(D, 2, dim=1, largest=False)

        arange = torch.arange(len(D))
        # rule1 spatial correspondence
        c1 = ind[:, 0] == arange.to(ind.device)
        # rule2 pass the ratio test
        c2 = val[:, 0] < val[:, 1] * content_thresh

        check = c2 * c1
        content_points.append(geo_points[i][check])
        affine_content_points.append(affine_geo_points[i][check])
    return content_points, affine_content_points


def geometric_filter(affine_detector_pred, points, affine_points, max_num=1024, geometric_thresh=0.5):
    """
    geometric matching in paper
    :param affine_detector_pred: geo_points probability of affine image
    :param points: nms results of input_image image
    :param affine_points: nms results of affine image
    :param max_num: maximum number of learned keypoints
    :param geometric_thresh: 
    :return: geometric-filtered keypoints
    """
    geo_points = []
    affine_geo_points = []
    for s, k in enumerate(affine_points):
        sample_aff_values = affine_detector_pred[s, 0, k[:, 1].long(), k[:, 0].long()]
        check = sample_aff_values.squeeze() >= geometric_thresh
        geo_points.append(points[s][check][:max_num])
        affine_geo_points.append(k[check][:max_num])

    return geo_points, affine_geo_points


def pke_learn(detector_pred, descriptor_pred, grid_inverse, affine_detector_pred,
              affine_descriptor_pred, kernel, loss_cal, label_point_positions,
              value_map, config, PKE_learn=True):
    """
    pke process used for detector
    :param detector_pred: probability map from raw image
    :param descriptor_pred: prediction of descriptor_pred network
    :param kernel: used for gaussian heatmaps
    :param mask_kernel: used for masking initial keypoints
    :param grid_inverse: used for inverse
    :param loss_cal: loss (default is dice)
    :param label_point_positions: positions of keypoints on labels
    :param value_map: value map for recoding and selecting learned geo_points
    :param pke_learn: whether to use PKE
    :return: loss of detector, num of additional geo_points, updated value maps and enhanced labels
    """
    # used for masking initial keypoints on enhanced labels
    initial_label = F.conv2d(label_point_positions, kernel,
                             stride=1, padding=(kernel.shape[-1] - 1) // 2)
    initial_label[initial_label > 1] = 1

    if not PKE_learn:
        return loss_cal(detector_pred, initial_label.to(detector_pred)), 0, None, None, initial_label

    nms_size = config['nms_size']
    nms_thresh = config['nms_thresh']
    scale = 8

    enhanced_label = None
    geometric_thresh = config['geometric_thresh']
    content_thresh = config['content_thresh']
    with torch.no_grad():
        h, w = detector_pred.shape[2:]

        # number of learned points
        number_pts = 0
        points = nms(detector_pred, nms_thresh=nms_thresh, nms_size=nms_size,
                     detector_label=initial_label, mask=True)

        # geometric matching
        points, affine_points = mapping_points(grid_inverse, points, h, w)
        geo_points, affine_geo_points = geometric_filter(affine_detector_pred, points, affine_points,
                                                         geometric_thresh=geometric_thresh)


        # content matching
        content_points, affine_contend_points = content_filter(descriptor_pred, affine_descriptor_pred, geo_points,
                                                               affine_geo_points, content_thresh=content_thresh,
                                                               scale=scale)
        enhanced_label_pts = []
        for step in range(len(content_points)):
            # used to combine initial points and learned points
            positions = torch.where(label_point_positions[step, 0] == 1)
            if len(positions) == 2:
                positions = torch.cat((positions[1].unsqueeze(-1), positions[0].unsqueeze(-1)), -1)
            else:
                positions = positions[0]

            final_points = update_value_map(value_map[step], content_points[step], config)

            # final_points = torch.cat((final_points, positions))

            temp_label = torch.zeros([h, w]).to(detector_pred.device)

            temp_label[final_points[:, 1], final_points[:, 0]] = 0.5
            temp_label[positions[:, 1], positions[:, 0]] = 1

            enhanced_kps = nms(temp_label.unsqueeze(0).unsqueeze(0), 0.1, 10)[0]
            if len(enhanced_kps) < len(positions):
                enhanced_kps = positions
            # print(len(final_points), len(positions), len(enhanced_kps))
            number_pts += (len(enhanced_kps) - len(positions))
            # number_pts += (len(enhanced_kps) - len(positions)) if (len(enhanced_kps) - len(positions)) > 0 else 0

            temp_label[:] = 0
            temp_label[enhanced_kps[:, 1], enhanced_kps[:, 0]] = 1

            enhanced_label_pts.append(temp_label.unsqueeze(0).unsqueeze(0))

            temp_label = F.conv2d(temp_label.unsqueeze(0).unsqueeze(0), kernel, stride=1,
                                  padding=(kernel.shape[-1] - 1) // 2)  # generating gaussian heatmaps
            temp_label[temp_label > 1] = 1

            if enhanced_label is None:
                enhanced_label = temp_label
            else:
                enhanced_label = torch.cat((enhanced_label, temp_label))

    enhanced_label_pts = torch.cat(enhanced_label_pts)
    affine_pred_inverse = F.grid_sample(affine_detector_pred, grid_inverse, align_corners=True)

    loss1 = loss_cal(detector_pred, enhanced_label)  # L_geo
    loss2 = loss_cal(detector_pred, affine_pred_inverse)  # L_clf
    # pred_mask = (enhanced_label > 0) & (affine_pred_inverse != 0)
    # loss2 = loss_cal(detector_pred[pred_mask], affine_pred_inverse[pred_mask])  # L_clf

    # mask_pred = grid_inverse
    # loss2 = loss_cal(detector_pred[mask_pred], affine_pred_inverse[mask_pred])  # L_clf

    loss = loss1+loss2

    return loss, number_pts, value_map, enhanced_label_pts, enhanced_label
