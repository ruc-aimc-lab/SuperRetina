import math
import os
import random
import shutil
import time

import torch
from torchvision import transforms as T
from torch.nn import functional as F
import numpy as np
import cv2
import scipy.stats as st
import config
from tqdm import tqdm
import matplotlib.pyplot as plt


def affine_images(images, used_for='detector'):
    """
    Perform affine transformation on images
    :param images: (B, C, H, W)
    :param keypoint_labels: corresponding labels
    :param value_map: value maps, used to record history learned geo_points
    :return: results of affine images, affine labels, affine value maps, affine transformed grid_inverse, inverse transformed grid_inverse
    """

    h, w = images.shape[2:]
    theta = None
    thetaI = None

    for i in range(len(images)):
        if used_for == 'detector':
            affine_params = T.RandomAffine(20).get_params(degrees=[-15, 15], translate=[0.2, 0.2],
                                                          scale_ranges=[0.9, 1.35],
                                                          shears=None, img_size=[h, w])
        else:
            affine_params = T.RandomAffine(20).get_params(degrees=[-3, 3], translate=[0.1, 0.1],
                                                          scale_ranges=[0.9, 1.1],
                                                          shears=None, img_size=[h, w])
        angle = -affine_params[0] * math.pi / 180
        theta_ = torch.tensor([
            [1 / affine_params[2] * math.cos(angle), math.sin(-angle), -affine_params[1][0] / images.shape[2]],
            [math.sin(angle), 1 / affine_params[2] * math.cos(angle), -affine_params[1][1] / images.shape[3]],
            [0, 0, 1]
        ], dtype=torch.float).to(images)
        thetaI_ = theta_.inverse()
        theta_ = theta_[:2]
        thetaI_ = thetaI_[:2]

        theta_ = theta_.unsqueeze(0)
        thetaI_ = thetaI_.unsqueeze(0)
        if theta is None:
            theta = theta_
        else:
            theta = torch.cat((theta, theta_))
        if thetaI is None:
            thetaI = thetaI_
        else:
            thetaI = torch.cat((thetaI, thetaI_))

    grid = F.affine_grid(theta, images.size(), align_corners=True)
    grid = grid.to(images)
    grid_inverse = F.affine_grid(thetaI, images.size(), align_corners=True)
    grid_inverse = grid_inverse.to(images)
    output = F.grid_sample(images, grid, align_corners=True)

    if used_for == 'descriptor':
        if random.random() >= 0.4:
            output = output.repeat(1, 3, 1, 1)
            output = T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.2)(output)
            output = T.Grayscale()(output)

    return output.detach().clone(), grid, grid_inverse


def get_gaussian_kernel(kernlen=21, nsig=5):
    """Get kernels used for generating Gaussian heatmaps"""
    interval = (2 * nsig + 1.) / kernlen
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)

    kern1d = np.diff(st.norm.cdf(x))

    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    weight = torch.nn.Parameter(data=kernel, requires_grad=False)
    weight = (weight - weight.min()) / (weight.max() - weight.min())
    return weight


def value_map_load(save_dir, names, input_with_label, shape=(768, 768), value_maps_running=None):
    value_maps = []
    for s, name in enumerate(names):
        path = os.path.join(save_dir, name.split('.')[0] + '.png')
        if input_with_label[s] and value_maps_running is not None and name in value_maps_running:
            value_map = value_maps_running[name]
        elif input_with_label[s] and value_maps_running is None and os.path.exists(path):
            value_map = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            value_map = np.zeros([shape[0], shape[1]]).astype(np.uint8)

        value_map = torch.from_numpy(value_map).unsqueeze(0).unsqueeze(0)
        value_maps.append(value_map)
    return torch.cat(value_maps)


def value_map_save(save_dir, names, input_with_label, value_maps, value_maps_running=None):
    for s, name in enumerate(names):
        if input_with_label[s]:
            vp = value_maps[s].squeeze().numpy()
            if value_maps_running is not None:
                value_maps_running[name] = vp
            else:
                path = os.path.join(save_dir, name.split('.')[0] + '.png')
                cv2.imwrite(path, vp)


def train_model(model, optimizer, dataloaders, device, num_epochs, train_config):
    model_save_path = train_config['model_save_path']
    model_save_epoch = train_config['model_save_epoch']
    pke_start_epoch = train_config['pke_start_epoch']
    pke_show_epoch = train_config['pke_show_epoch']
    pke_show_list = train_config['pke_show_list']
    is_value_map_save = train_config['is_value_map_save']
    value_map_save_dir = train_config['value_map_save_dir']

    if is_value_map_save:
        if os.path.exists(value_map_save_dir):
            shutil.rmtree(value_map_save_dir)
        os.makedirs(value_map_save_dir)

    value_maps_running = None
    if not is_value_map_save:
        value_maps_running = {}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch may have some phases
        phases = list(dataloaders.keys())
        model.PKE_learn = True
        if epoch < pke_start_epoch:
            model.PKE_learn = False
        for phase in phases:
            pke_show = train_config['pke_show']
            image_shows = []
            init_kp_shows = []
            label_shows = []
            enhanced_kp_shows = []
            show_names = []

            if 'val' in phase:
                if epoch % model_save_epoch == 0:
                    print(f'save model for epoch {epoch}')
                    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                    torch.save(state, model_save_path)
                continue
            print('-' * 10 + 'phase:' + phase + '\t PKE_learn:' + str(model.PKE_learn) + '-' * 10)

            if 'train' in phase:
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            epoch_samples = 0
            print_descriptor_loss = 0
            print_detector_loss = 0
            num_learned_pts = 0
            num_input_with_label = 0
            num_input_descriptor = 0

            for images, input_with_label, keypoint_positions, label_names \
                    in tqdm(dataloaders[phase]):
                batch_size = images.shape[0]
                learn_index = torch.where(input_with_label)

                images = images.to(device)

                value_maps = value_map_load(value_map_save_dir, label_names, input_with_label,
                                            images.shape[-2:], value_maps_running)

                value_maps = value_maps.to(device)
                keypoint_positions = keypoint_positions.to(device)

                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled('train' in phase):

                    loss, number_pts_one, print_loss_detector_one, print_loss_descriptor_one, enhanced_label_pts, \
                        enhanced_label, detector_pred, loss_detector_num, loss_descriptor_num \
                        = model(images, keypoint_positions, value_maps, learn_index)
                    if enhanced_label_pts is None:
                        enhanced_label_pts = keypoint_positions
                    if pke_show:
                        if len(pke_show_list) == 0 and len(learn_index[0]) != 0:  # randomly show one image
                            show_names = [(label_names[learn_index[0][0]], learn_index[0][0])]
                            pke_show = False
                        for s, label_path in enumerate(label_names):
                            file_name = os.path.split(label_path)[-1].split('.')[0]
                            if file_name in pke_show_list:
                                show_names.append((label_path, s))

                        for (show_name, idx) in show_names:
                            image_shows.append(images[idx][0].cpu().data)
                            init_kp_shows.append(keypoint_positions[idx][0].cpu().data)
                            label_shows.append(enhanced_label[idx][0].cpu().data)
                            pred_show = detector_pred.detach().cpu()[idx][0]
                            enhanced_kp_shows.append(enhanced_label_pts[idx][0].cpu().data)

                    print_detector_loss += print_loss_detector_one * len(learn_index[0])
                    print_descriptor_loss += print_loss_descriptor_one * batch_size
                    num_input_with_label += loss_detector_num

                    num_learned_pts += number_pts_one

                    if 'train' in phase:
                        loss.backward()
                        optimizer.step()
                # update value map
                if len(learn_index[0]) != 0:
                    value_maps = value_maps.cpu()
                    value_map_save(value_map_save_dir, label_names, input_with_label, value_maps, value_maps_running)

                # statistics
                num_input_descriptor += loss_descriptor_num
                epoch_samples += batch_size

                ###################################################################################
            print_detector_loss = print_detector_loss / num_input_with_label
            print_descriptor_loss = print_descriptor_loss / epoch_samples
            print(phase, 'overall loss: {}'.format(print_detector_loss + print_descriptor_loss),
                  'detector_loss: {} of {} nums, #avg learned keypoints:{} '.format(print_detector_loss,
                                                                                    num_input_with_label,
                                                                                    num_learned_pts / num_input_with_label),
                  'descriptor_loss: {} of {} nums'.format(print_descriptor_loss, num_input_descriptor))

            for s, (name, _) in enumerate(show_names):
                if not epoch % pke_show_epoch == 0:
                    break
                input_show = image_shows[s]
                label_show = label_shows[s]
                init_kp_show = init_kp_shows[s]
                enhanced_kp_show = enhanced_kp_shows[s]
                name = os.path.split(name)[-1].split('.')[0]

                plt.figure(dpi=100)
                plt.imshow(input_show, 'gray')
                plt.title('epoch:{}, phase:{}, name:{}'.format(epoch, phase, name))
                plt.axis('off')
                try:
                    y, x = torch.where(enhanced_kp_show.cpu() == 1)
                    plt.scatter(x, y, s=2, c='springgreen')
                except Exception:
                    pass
                try:
                    y, x = torch.where(init_kp_show.cpu() == 1)
                    plt.scatter(x, y, s=2, c='b')
                except Exception:
                    pass
                # plt.savefig(f'./data/draw/train_new_new/{name1}_epo{epoch}_{len(x)}.png',bbox_inches='tight',pad_inches = -0.1)
                plt.show()
                plt.figure(dpi=200)
                plt.subplot(121)
                plt.imshow(pred_show, 'gray')
                plt.subplot(122)
                plt.imshow(label_show, 'gray')
                plt.show()
                plt.close()
                plt.pause(0.1)
