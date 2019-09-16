"""
Based on the source code from these links
https://github.com/timgaripov/swa
https://github.com/izmailovpavel/contrib_swa_examples
"""
from comet_ml import Experiment
import argparse
import datetime
import os
import random
import sys
import time

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
import torchcontrib
import torchsummary
from torch import nn
from torchvision import transforms
from tqdm import tqdm

import utils
from utils import OpenEDS, Rescale, ToTensor, Normalize, Brightness


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


""" ConvBNReLU"""


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


""" ConvReLU"""


class ConvReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


""" InvertedResidual """


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MobileNetV2_CS(nn.Module):
    def __init__(self, num_classes=4, out_shape=(640, 400), width_mult=1.0, inverted_residual_setting=None,
                 round_nearest=8):
        """
        MobileNet V2 CS main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2_CS, self).__init__()
        block = InvertedResidual
        input_channel = 32
        self.out_shape = out_shape
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 2, 1],
                [6, 24, 3, 2],
                [6, 32, 4, 2],
                # [6, 64, 4, 2]
                # [1, 16, 1, 1],
                # [6, 24, 2, 2],
                # [6, 32, 3, 2],
                # [6, 64, 4, 2],
                # [6, 96, 3, 1],
                # [6, 160, 3, 2],
                # [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)

        features = [ConvBNReLU(1, input_channel, stride=2)]  # 3 for color image

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # building last several layers
        kn_size = 3
        features.append(ConvReLU(input_channel, 64, kernel_size=kn_size))
        # features.append(ConvReLU(64, 64, kernel_size=kn_size))

        self.features = nn.Sequential(*features)
        # building segmentation layer (1x1)
        # self.aspp = ASPP(in_channels=64, atrous_rates=[16, 24, 32])
        # self.aspp = ASPP(in_channels=64, atrous_rates=[6, 12, 18])
        c_segmentation = [64, num_classes]

        # x_up = [ConvReLU(c_segmentation[0], c_segmentation[1], kernel_size=1),
        #         nn.Upsample(scale_factor=16.0, mode='bilinear', align_corners=False)]
        # self.x_up = nn.Sequential(*x_up)

        segmentation_part1 = [ConvReLU(c_segmentation[0], c_segmentation[0], kernel_size=1),
                              nn.Upsample(scale_factor=4.0, mode='bilinear',
                                          align_corners=False)]

        up_part1 = [ConvReLU(c_segmentation[0], c_segmentation[1], kernel_size=1),
                    nn.Upsample(scale_factor=4.0, mode='bilinear', align_corners=False),
                    SELayer(channel=c_segmentation[1], reduction=4)]

        self.up_part1 = nn.Sequential(*up_part1)

        conv_up = [ConvReLU(c_segmentation[0], c_segmentation[1], kernel_size=kn_size),
                   ConvReLU(c_segmentation[1], c_segmentation[1], kernel_size=kn_size),
                   ConvReLU(c_segmentation[1], c_segmentation[1], kernel_size=kn_size),
                   nn.Upsample(scale_factor=4.0, mode='bilinear', align_corners=False)]
        self.conv_up_part1 = nn.Sequential(*conv_up)

        self.segm_part1 = nn.Sequential(*segmentation_part1)

        # self.alpha = nn.Parameter(torch.tensor([0.7]))
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)

        # x_aspp = self.aspp(x)
        # x_aspp_up = F.interpolate(x_aspp, scale_factor=4.0, mode='bilinear', align_corners=True)
        # print(x_aspp_up.shape)
        x1 = self.segm_part1(x)

        # x1_aspp = torch.cat([x_aspp_up, x1], dim=1)

        # x1_seg = self.conv_up_part1(x1_aspp)
        x1_seg = self.conv_up_part1(x1)

        x1_up = self.up_part1(x1)
        # alpha = 0.7
        # x = x1_seg * self.alpha.expand_as(x1_seg) + x1_up * torch.sub(1.0, self.alpha).expand_as(x1_up)

        # x_up_ = self.x_up(x)
        x = x1_seg + x1_up  # + x_up_

        x_softmax = F.softmax(x, dim=1)
        sgm = torch.argmax(x_softmax, dim=1)
        return x_softmax, sgm


def weighted_CrossEntropyLoss(output, target, device, n_classes=4):
    """ Weighted Cross Entropy Loss"""
    n_pixel = target.numel()
    _, counts = torch.unique(target, return_counts=True)
    cls_weight = torch.div(n_pixel, n_classes * counts.type(torch.FloatTensor)).to(device)
    loss = F.cross_entropy(output, target, weight=cls_weight)

    return loss


def get_mean_std(data_loader, device):
    """ Get mean, std of data_loader """
    print('Calculating mean, std ...')
    cnt = 0
    fst = torch.empty(3).to(device)  # 0
    snd = torch.empty(3).to(device)  # 0
    for i_batch, sample_batched in enumerate(tqdm(data_loader)):
        img_batch = sample_batched['image'].to(device)
        b, c, h, w = img_batch.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(img_batch, dim=[0, 2, 3])
        sum_of_square = torch.sum(torch.pow(img_batch, 2.0), dim=[0, 2, 3])
        fst = (sum_ + cnt * fst) / (nb_pixels + cnt)
        snd = (sum_of_square + cnt * snd) / (nb_pixels + cnt)

        cnt += nb_pixels

    return fst, torch.sqrt(snd - torch.pow(fst, 2.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eye Segmentation - SGD/SWA training')
    parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
    parser.add_argument('--opt', type=str, default=None, required=True, help='Optimizer: Adam or SGD (default: None)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size (default: 128)')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
    parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
    parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
    parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR',
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
    parser.add_argument('--swa_start', type=float, default=161, metavar='N',
                        help='SWA start epoch number (default: 161)')
    parser.add_argument('--swa_lr', type=float, default=0.0005, metavar='LR', help='SWA LR (default: 0.05)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()

    experiment = Experiment(project_name='OpenEDS_MODIFIED', api_key='uG1BcicYOr83KvLjFEZQMrWVg',
                            auto_output_logging='simple',
                            disabled=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Preparing directory %s' % args.dir)
    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
        f.write(experiment.get_key())
        f.write('\n')

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print('Loading dataset. ', str(datetime.datetime.now()))
    # root_path = './Semantic_Segmentation_Dataset/'
    if os.name == 'nt':
        root_path = 'C:/Semantic_Segmentation_Dataset/'
    else:
        root_path = './Semantic_Segmentation_Dataset/'

    cfg = dict()
    cfg['batch_size'] = 64

    cfg['scale'] = 0.5
    # Original, mean 0.4679, std 0.2699
    # Gamma correction: mean 0.3977, std 0.2307
    if cfg['scale'] == 0.5:
        mnet_v2_mean = [0.4679]
        mnet_v2_std = [0.2699]
    else:
        mnet_v2_mean = [0.4679]
        mnet_v2_std = [0.2699]

    train_set = OpenEDS(root_path=root_path + 'train',
                        transform=transforms.Compose(
                            [Rescale(cfg['scale']), Brightness(brightness=(0.5, 2.75)), ToTensor(),
                             Normalize(mnet_v2_mean, mnet_v2_std)]))

    val_set = OpenEDS(root_path=root_path + 'validation',
                      transform=transforms.Compose(
                          [Rescale(cfg['scale']), ToTensor(), Normalize(mnet_v2_mean, mnet_v2_std)]))  #

    test_set = OpenEDS(root_path=root_path + 'test',
                       transform=transforms.Compose(
                           [Rescale(cfg['scale']), ToTensor(), Normalize(mnet_v2_mean, mnet_v2_std)]))  #

    loaders = {'train': torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.num_workers, pin_memory=True),
               'val': torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True),
               'test': torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.num_workers, pin_memory=True)
               }

    experiment.log_dataset_hash(loaders)
    # train_mean, train_std = get_mean_std(loaders['train'], device)
    # print('Mean {}, STD {}'.format(train_mean, train_std))
    # sys.exit(0)

    # print('Train: {}. Val: {}. Test: {}'.format(len(train_set), len(val_set), len(test_set)))
    # for dtt in ['train', 'val', 'test']:
    #     loader = loaders[dtt]
    #     print(dtt, len(loader))
    #     for i, sample_batched in enumerate(tqdm(loader)):
    #         input = sample_batched['image']
    #
    # sys.exit(0)

    model = MobileNetV2_CS()
    model.to(device)
    criterion = utils.generalised_dice_loss_ce  # weighted_CrossEntropyLoss
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.wd)
    torchsummary.summary(model, (1, 320, 200))


    # sys.exit(0)

    def schedule(epoch):
        t = (epoch) / (args.swa_start if args.swa else args.epochs)
        lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
        if args.lr_init > 5e-1:
            t_threshold = 0.16
        # elif args.lr_init > 1e-3:
        #     t_threshold = 0.4
        else:
            t_threshold = 0.5
        if t <= t_threshold:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - t_threshold) / (t_threshold - 0.1)
            if factor <= 0:
                factor = lr_ratio
        else:
            factor = lr_ratio
        return args.lr_init * factor


    def our_schedule(epoch):

        factor = (epoch + 1) // 10 + 1
        return args.lr_init / factor


    if args.swa:
        print('SWA training')
        steps_per_epoch = len(loaders['train'].dataset) / args.batch_size
        steps_per_epoch = int(steps_per_epoch)
        print("Steps per epoch:", steps_per_epoch)
        optimizer = torchcontrib.optim.SWA(optimizer, swa_start=args.swa_start * steps_per_epoch,
                                           swa_freq=steps_per_epoch, swa_lr=args.swa_lr)
    else:
        print('Original training')

    start_epoch = 0
    if args.resume is not None:
        print('Resume training from {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'tr_mIOU', 'val_loss', 'val_acc', 'val_mIOU', 'time']
    if args.swa:
        columns = columns[:-1] + ['swa_val_loss', 'swa_val_acc', 'swa_val_mIOU'] + columns[-1:]
        swa_res = {'loss': None, 'accuracy': None, 'mIOU': None}

    utils.save_checkpoint(
        args.dir,
        start_epoch,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )

    if args.resume is not None:
        if args.swa:
            optimizer.swap_swa_sgd()

    # print('Alpha value: ', model.alpha)
    """ Training """
    for epoch in range(start_epoch, args.epochs):
        time_ep = time.time()

        # if args.swa:
        lr = schedule(epoch)
        # else:
        #     lr = our_schedule(epoch)

        utils.adjust_learning_rate(optimizer, lr)
        train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer)
        experiment.log_metric("learning_rate", lr)

        # Log train_res
        with experiment.train():
            experiment.log_metrics(train_res, step=epoch)

        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            val_res = utils.eval(loaders['val'], model, criterion, viz='val_{}'.format(epoch + 1))
            # Log val_res
            with experiment.validate():
                experiment.log_metrics(val_res, step=epoch)
        else:
            val_res = {'loss': None, 'accuracy': None, 'mIOU': None}

        if args.swa and (epoch + 1) >= args.swa_start:
            if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
                # Batchnorm update
                print('BatchNorm Update')
                optimizer.swap_swa_sgd()
                optimizer.bn_update(loaders['train'], model, device)
                swa_res = utils.eval(loaders['val'], model, criterion, viz='swa_val_{}'.format(epoch + 1))

                if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
                    utils.save_checkpoint(
                        args.dir,
                        epoch + 1,
                        state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict()
                    )

                optimizer.swap_swa_sgd()
                with experiment.validate():
                    swa_res_log = {'swa_loss': swa_res['loss'], 'swa_accuracy': swa_res['accuracy'],
                                   'swa_mIOU': swa_res['mIOU']}
                    experiment.log_metrics(swa_res_log, step=epoch)
            else:
                swa_res = {'loss': None, 'accuracy': None, 'mIOU': None}

        if (epoch + 1) % args.save_freq == 0:
            if args.swa is None or (args.swa and (epoch + 1) < args.swa_start):
                utils.save_checkpoint(
                    args.dir,
                    epoch + 1,
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict()
                )

        time_ep = time.time() - time_ep
        values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], train_res['mIOU'], val_res['loss'],
                  val_res['accuracy'], val_res['mIOU'], time_ep]
        if args.swa:
            values = values[:-1] + [swa_res['loss'], swa_res['accuracy'], swa_res['mIOU']] + values[-1:]

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.6f')
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
    """ End of Training """

    # if args.epochs % args.save_freq != 0:
    #     optimizer.swap_swa_sgd()
    #     optimizer.bn_update(loaders['train'], model, device)
    #
    #     utils.save_checkpoint(
    #         args.dir,
    #         args.epochs,
    #         state_dict=model.state_dict(),
    #         optimizer=optimizer.state_dict()
    #     )

    print('End of training, ', str(datetime.datetime.now()))
    # print('Val results: ', utils.eval(loaders['val'], model, criterion), device)
    # utils.test_writer(model, device, loaders['test'], write_folder=root_path + 'test/predicts/')
    # print('Alpha value: ', model.alpha)
    experiment.end()
