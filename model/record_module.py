import torch

from common.common_util import simple_nms


def update_value_map(value_map, points, value_map_config):
    """
    Update value maps used for recording learned keypoints from PKE,
    and getting the final learned keypoints which are combined of previous learned keypoints.
    :param value_map: previous value maps
    :param points: the learned keypoints in this epoch
    :param value_map_config:
    :return: the final learned keypoints combined of previous learning points
    """

    raw_value_map = value_map.clone()
    # used for record areas of value=0
    raw_value_map[value_map == 0] = -1
    area_set = value_map_config['area']
    area = area_set // 2

    value_increase_point = value_map_config['value_increase_point']
    value_increase_area = value_map_config['value_increase_area']
    value_decay = value_map_config['value_decay']

    h, w = value_map[0].shape
    for (x, y) in points:
        y_d = y - area // 2 if y - area // 2 > 0 else 0
        y_u = y + area // 2 if y + area // 2 < h else h
        x_l = x - area // 2 if x - area // 2 > 0 else 0
        x_r = x + area // 2 if x + area // 2 < w else w
        tmp = value_map[0, y_d:y_u, x_l:x_r]
        if value_map[0, y, x] != 0 or tmp.sum() == 0:
            value_map[0, y, x] += value_increase_point  # if there is no learned point before, then add a high value
        else:
            tmp[tmp > 0] += value_increase_area
            value_map[0, y_d:y_u, x_l:x_r] = tmp

    value_map[torch.where(
        value_map == raw_value_map)] -= value_decay  # value decay of positions that don't appear this time

    tmp = value_map.detach().clone()

    tmp = simple_nms(tmp.unsqueeze(0).float(), area_set*2)
    tmp = tmp.squeeze()

    final_points = torch.nonzero(tmp >= value_increase_point)
    final_points = torch.flip(final_points, [1]).long()  # to x, y
    return final_points
