import argparse
import gc
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchsummary
from scipy import signal
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from skimage import measure
from torchvision import transforms
from tqdm import tqdm

import utils
from train import MobileNetV2_CS
from utils import OpenEDS, Rescale, ToTensor, Normalize


def eye_refinement(img, img_name=None):
    """
    Eye Refinement
    :param img:
    :param img_name:
    :return:
    """

    def re_label(img, value, ipf=None):
        """
        Re-label img with value (0, 1, 2)
        :param img:
        :param value:
        :param ipf:
        :return:
        """

        def fill_holes(img, expr=lambda x: x > 0):
            img_binary = 255 * expr(img).astype(np.uint8)
            img_tmp = img_binary.copy()
            h, w = img_tmp.shape
            mask = np.zeros((h + 2, w + 2), np.uint8)

            cv2.floodFill(img_tmp, mask, (0, 0), 255)
            cv2.floodFill(img_tmp, mask, (0, 639), 255)
            img_fill = np.bitwise_or(img_binary, ~img_tmp)

            return img_fill

        img_filled = fill_holes(img, lambda x: x > value)

        # Remove small regions (not eye region)
        mask_regions, n_regions = measure.label(img_filled, connectivity=2, return_num=True)
        ld_area = -1
        idx = 0
        prev_idx = 0

        for props in measure.regionprops(mask_regions, img):
            idx = idx + 1
            if props.area > ld_area and (value > 0 or (value == 0 and props.max_intensity > 1)):
                ld_area = props.area
                img_filled[mask_regions == prev_idx] = 0
                prev_idx = idx
            else:
                img_filled[mask_regions == idx] = 0

        rf = ipf
        if value == 0:
            # Remove false region with peaks analysis
            horizontal_projection = np.sum(img_filled > 0, axis=1)
            peakind, _ = signal.find_peaks(horizontal_projection, distance=img_filled.shape[1] / 8)
            if len(peakind) >= 2:
                fp = -1
                if horizontal_projection[peakind[0]] > horizontal_projection[peakind[1]]:
                    fp = peakind[0]
                    sp = peakind[1]
                elif len(peakind) >= 3 and horizontal_projection[peakind[1]] > horizontal_projection[peakind[2]]:
                    fp = peakind[1]
                    sp = peakind[2]

                if fp > -1:
                    clr_index = np.argmin(horizontal_projection[fp: sp]) + fp
                    img_filled[clr_index:, :] = 0

            _, contours, _ = cv2.findContours((img_filled > 0).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)
            x_index = cnt[hull[1:, 0], :, 0].flatten()
            end_index = np.argwhere(x_index[1:] < x_index[:-1])[0][0] + 1
            line_below_index = np.arange(hull[0, 0], hull[end_index, 0])
            emid = np.argwhere(hull[end_index + 1:, 0] < hull[end_index:-1, 0])[0][0] + end_index
            line_above_index = np.hstack(
                [np.arange(hull[end_index, 0], hull[emid, 0]), np.arange(hull[emid + 1, 0], hull[-1, 0]), np.arange(hull[-1, 0], hull[0, 0])])
            x_below = contours[0][line_below_index, :, 0].flatten()
            pk_below, _ = signal.find_peaks(x_below, width=img_filled.shape[1] * 1 / 20)

            if len(pk_below) > 0:
                cnt_refined = np.vstack(
                    [contours[0][line_below_index[:pk_below[0]], :, :], contours[0][line_above_index, :, :]])
                cnt_mask = np.zeros(img_filled.shape, dtype=np.uint8)
                cv2.drawContours(cnt_mask, [cnt_refined], -1, (255), -1)
                cnt_mask[:, np.max(x_below[:pk_below[0]]):] = 0
                img_filled[(255 - cnt_mask).astype(np.bool)] = 0

                print(img_name)

                _, contours, _ = cv2.findContours((img_filled > value).astype(np.uint8), cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
                cnt = contours[0]
                hull = cv2.convexHull(cnt, returnPoints=False, clockwise=True)
                x_index = cnt[hull[1:, 0], :, 0].flatten()
                end_index = np.argwhere(x_index[1:] < x_index[:-1])[0][0] + 1
                line_below_index = np.arange(hull[0, 0], hull[end_index, 0])
                emid = np.argwhere(hull[end_index + 1:, 0] < hull[end_index:-1, 0])[0][0] + end_index
                line_above_index = np.hstack(
                    [np.arange(hull[end_index, 0], hull[emid, 0]), np.arange(hull[emid + 1, 0], hull[-1, 0]), np.arange(hull[-1, 0], hull[0, 0])])

            y_above = contours[0][line_above_index, :, 1].flatten()
            pk_above, _ = signal.find_peaks(y_above, width=img_filled.shape[1] * 1 / 40)
            # if img_name == '000000347521':
            #     print('***', img_name)
            #     print(len(pk_above))
            if len(pk_above) > 0:
                print(img_name)
                x_above = contours[0][line_above_index, :, 0].flatten()

                vs = ConvexHull(np.asarray([x_above, y_above]).transpose()).vertices
                indices_of_upper_hull_verts = list(
                    reversed(np.concatenate([vs[np.where(vs == len(x_above) - 1)[0][0]:], vs[0:1]])))
                newX = x_above[indices_of_upper_hull_verts]
                newY = y_above[indices_of_upper_hull_verts]
                x_smooth = np.arange(newX.max(), newX.min(), -1)
                f = interp1d(newX, newY, kind='quadratic')
                y_smooth = f(x_smooth)

                above_refined = np.vstack([contours[0][line_below_index, :, :],
                                           np.vstack([x_smooth, y_smooth]).T.reshape(-1, 1, 2).astype(np.int64)])

                above_mask = np.zeros(img_filled.shape, dtype=np.uint8)
                cv2.drawContours(above_mask, [above_refined], -1, 255, -1)
                ipf = np.logical_and(above_mask, ~(img_filled > 0))
                img_filled[ipf] = value + 1

                rf = ipf

        elif ipf is not None:
            print(img_name)
            _, ip_cnt, _ = cv2.findContours((img_filled > value).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(ip_cnt) > 0:
                (xc, yc), rc = cv2.minEnclosingCircle(ip_cnt[0])
                ip_mask = np.zeros(img_filled.shape, dtype=np.uint8)
                cv2.circle(ip_mask, (int(xc), int(yc)), int(rc), 255, -1)

                img_filled[np.logical_and(ip_mask > 0, ipf)] = value + 1

                img_filled = fill_holes(img_filled, lambda x: x > value)
            rf = img_filled > value

        return img_filled, rf

    img_filled_sclera, ripf = re_label(img, value=0, ipf=None)
    img_sclera = img.copy()
    img_sclera[img_filled_sclera == 0] = 0  # Largest region, consider as eye

    img_filled_iris, rpf = re_label(img_sclera, value=1, ipf=ripf)
    img_iris = img_sclera.copy()
    img_iris[img_filled_iris == 0] = 0

    img_filled_pupil, _ = re_label(img_iris, value=2, ipf=rpf)
    img_pupil = img_sclera.copy()
    img_pupil[img_filled_pupil] = 0

    img_ret = np.zeros(img.shape, dtype=np.uint8)
    img_ret[img_filled_sclera > 0] = 1
    img_ret[img_filled_iris > 0] = 2
    img_ret[img_filled_pupil > 0] = 3

    n_part = np.unique(img_ret)
    if n_part.size < 4:
        # print(img_name)
        if 1 not in n_part:
            if 2 in n_part:
                img_ret[img_ret == 2] = 1
                if 3 in n_part:
                    img_ret[img_ret == 3] = 2
            elif 3 in n_part:
                img_ret[img_ret == 3] = 1
        else:
            if 2 not in n_part:
                if 3 in n_part:
                    img_ret[img_ret == 3] = 2

    return img_ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eye Segmentation - SGD/SWA Testing')
    # parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
    parser.add_argument('--checkpoint', type=str, default=None, required=True, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')
    parser.add_argument('--checkpoint_2', type=str, default=None, required=False, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')

    args = parser.parse_args()

    print('Loading dataset.')
    root_path = './Semantic_Segmentation_Dataset/'
    write_folder = root_path + 'test/predicts/'
    os.makedirs(write_folder, exist_ok=True)
    os.makedirs(write_folder + 'imgs/', exist_ok=True)
    with open(os.path.join(write_folder, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    cfg = dict()
    cfg['batch_size'] = 64

    cfg['scale'] = 0.5
    if cfg['scale'] == 0.5:
        mnet_v2_mean = [0.4679]
        mnet_v2_std = [0.2699]
    else:
        mnet_v2_mean = [0.4679]
        mnet_v2_std = [0.2699]

    train_set = OpenEDS(root_path=root_path + 'train',
                        transform=transforms.Compose(
                            [Rescale(cfg['scale']), ToTensor(), Normalize(mnet_v2_mean, mnet_v2_std)]))  #
    val_set = OpenEDS(root_path=root_path + 'validation',
                      transform=transforms.Compose(
                          [Rescale(cfg['scale']), ToTensor(), Normalize(mnet_v2_mean, mnet_v2_std)]))  #

    test_set = OpenEDS(root_path=root_path + 'test',
                       transform=transforms.Compose(
                           [Rescale(cfg['scale']), ToTensor(), Normalize(mnet_v2_mean, mnet_v2_std)]))  #

    loaders = {'train': torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers),
               'val': torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=False),
               'test': torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.num_workers, pin_memory=False)
               }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2_CS()
    model.to(device)

    print('Load model from {}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    print('Val set: ', utils.eval(loaders['val'], model, utils.generalised_dice_loss_ce))
    torch.save(model.state_dict(), write_folder + 'model_state_dict.pt')
    torch.save(model, write_folder + 'model_full.pt')
    loader = loaders['test']
    # sys.exit(0)
    # tmp = np.load(write_folder + '000000337010.npy')
    # tmp_ref = eye_refinement(tmp)
    #
    # sys.exit(0)

    if args.checkpoint_2:
        model_2 = MobileNetV2_CS()
        model_2.to(device)
        checkpoint_2 = torch.load(args.checkpoint_2)
        model_2.load_state_dict(checkpoint_2['state_dict'])
        model_2.eval()
        print('Val set M2: ', utils.eval(loaders['val'], model_2, utils.generalised_dice_loss_ce))
        alpha = 0.75
    else:
        model_2 = None
        alpha = 0

    # print('Alpha value: ', model.alpha)
    for i, sample_batched in enumerate(tqdm(loader)):
        gc.collect()
        input = sample_batched['image'].to(device)
        target = sample_batched['mask'].to(device)

        out_probs, out_cat = model(input)
        if model_2:
            out_probs_2, _ = model_2(input)
            out_probs_avg = (1.0 - alpha) * out_probs + alpha * out_probs_2
            out_cat = torch.argmax(out_probs_avg, dim=1)

        img_names = sample_batched['name']
        img_mask_predict = out_cat.cpu().numpy().astype(np.uint8)
        for idx in range(img_mask_predict.shape[0]):
            cur_predict = img_mask_predict[idx, :, :]
            cur_name = img_names[idx]
            # np.save(write_folder + cur_name, cur_predict)
            # if cur_name == '000000337055':
            #     print('Stop here')
            # else:
            #     continue

            # if cur_name in ['000000337742', '000000350291', '000000353150']:
            #     plt.imsave('E:/' + cur_name + '_pred.png', cur_predict)
            #     cur_predict_ref = eye_refinement(cur_predict, cur_name)
            #     plt.imsave('E:/' + cur_name + '_pred_ref.png', cur_predict_ref)
            # else:
            #     continue

            fig = plt.figure()
            ax1 = fig.add_subplot(131)
            ax1.imshow(cur_predict)
            ax1.set_title('Predicted')

            np.save(write_folder + cur_name + '_orig', cur_predict)

            cur_predict_ref = eye_refinement(cur_predict, cur_name)
            np.save(write_folder + cur_name, cur_predict_ref)
            ax2 = fig.add_subplot(132)
            ax2.imshow(cur_predict_ref)
            ax2.set_title('Refine Predicted')

            ax3 = fig.add_subplot(133)
            ax3.imshow(input.cpu().numpy()[idx, 0, :, :], cmap='gray')
            ax3.set_title('Image')

            plt.savefig('{}{}{}.png'.format(write_folder, 'imgs/', cur_name))
            plt.clf()
            plt.close()
            del fig

        #     if cur_name == '000000347521':
        #         break
        #
        # if cur_name == '000000337039':
        #     break
    # print('Alpha value: ', model.alpha)
    model.train()
    torchsummary.summary(model, (1, 320, 200))
