import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchsummary
from PIL import Image
from skimage import measure
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm


# __all__ = ['OpenEDS']


class ToTensor(object):
    """ Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        img, mask, name = np.asarray(sample['image']), sample['mask'], sample['name']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # img is gray scale image
        img = img[np.newaxis, :] / 255.0
        # img = img.transpose((2, 0, 1)) / 255.0
        return {'image': torch.from_numpy(img).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).type(torch.LongTensor), 'name': name}


class Rescale(object):
    """ Rescale the image in a sample to a given size """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, float, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img, mask, name = sample['image'], sample['mask'], sample['name']
        w, h = img.size  # PIL image, size return w, h
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        elif isinstance(self.output_size, float):
            new_h, new_w = self.output_size * h, self.output_size * w
        else:
            new_h, new_w = self.output_size

        new_h = int(new_h)
        new_w = int(new_w)
        img = transforms.Resize((new_h, new_w))(img)

        return {'image': img, 'mask': mask, 'name': name}


class Brightness(object):
    """ Rescale the image in a sample to a given size """

    def __init__(self, brightness):
        self.brightness = brightness

    def __call__(self, sample):
        img, mask, name = sample['image'], sample['mask'], sample['name']

        img = transforms.ColorJitter(brightness=self.brightness)(img)

        return {'image': img, 'mask': mask, 'name': name}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, mask, name = sample['image'], sample['mask'], sample['name']
        img = transforms.Normalize(mean=self.mean, std=self.std)(img)
        return {'image': img, 'mask': mask, 'name': name}


class OpenEDS(data.Dataset):
    def __init__(self, root_path, transform=None):
        """ Initialization """
        self.list_png = sorted(glob.glob(root_path + '/images/*.png'))
        self.transform = transform

    def __len__(self):
        """ Denotes the toal number of samples """
        return len(self.list_png)

    @staticmethod
    def adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def __getitem__(self, index):
        img_path = self.list_png[index]
        npy_path = img_path.replace('images', 'labels').replace('png', 'npy')

        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        # img = io.imread(img_path, as_gray=True)
        # img = Image.open(img_path.replace('png', 'bmp')).convert('L')
        img = Image.open(img_path)#.convert('L')  # Original images
        if img.mode == 'RGB':
            img = img.convert('L')

        if 'test' in img_path:
            npy = np.array([])
        else:
            npy = np.load(npy_path, allow_pickle=False)

        img_name = img_path.replace('\\', '/').split('/')[-1][:-4]
        sample = {'image': img, 'mask': npy, 'name': img_name}
        if self.transform:
            sample = self.transform(sample)

        return sample


""" Define metrics """


def pixel_acc(pred, label):
    acc = torch.eq(pred, label).type(torch.FloatTensor).mean()
    return acc


def mean_iou(pred, label, num_classes=4):
    """
    Mean IoU
    :param pred:
    :param label:
    :param num_classes:
    :return:
    """
    accum = False
    iou = None
    for idx in range(num_classes):
        out1 = (pred == idx)
        out2 = (label == idx)

        intersect = torch.sum(out1 & out2, dim=(1, 2)).type(torch.FloatTensor)
        union = torch.sum(out1 | out2, dim=(1, 2)).type(torch.FloatTensor)
        if accum:
            iou = iou + torch.div(intersect, union + 1e-16)
        else:
            iou = torch.div(intersect, union + 1e-16)
            accum = True
    m_iou = torch.mean(iou) / num_classes
    return m_iou


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)


def visualize(pred, label, idx=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(pred)
    ax1.set_title('Predicted')

    ax2 = fig.add_subplot(122)
    ax2.imshow(label)
    ax2.set_title('Ground truth')

    if idx is not None:
        os.makedirs('./viz/', exist_ok=True)
        plt.savefig('./viz/{}.png'.format(idx))
        plt.clf()
        plt.close()
        del fig


def train_epoch(loader, model, criterion, optimizer, device='cuda:0'):
    loss_sum = 0.0
    eval_pixel_acc = []
    eval_mean_iou = []

    model.train()

    for i, sample_batched in enumerate(loader):
        input = sample_batched['image'].to(device)
        target = sample_batched['mask'].to(device)

        optimizer.zero_grad()

        out_probs, out_cat = model(input)
        loss = criterion(out_probs, target, device=device)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        eval_pixel_acc.append(pixel_acc(out_cat, target))
        eval_mean_iou.append(mean_iou(out_cat, target))

    eval_pixel_acc = np.mean(eval_pixel_acc)
    eval_mean_iou = np.mean(eval_mean_iou)
    return {
        'loss': loss_sum,
        'accuracy': eval_pixel_acc,
        'mIOU': eval_mean_iou
    }


def eval(loader, model, criterion, device='cuda:0', viz=None):
    loss_sum = 0.0
    eval_pixel_acc = []
    eval_mean_iou = []

    model.eval()

    for i, sample_batched in enumerate(loader):
        input = sample_batched['image'].to(device)
        target = sample_batched['mask'].to(device)

        out_probs, out_cat = model(input)
        loss = criterion(out_probs, target, device=device)

        loss_sum += loss.item()
        eval_pixel_acc.append(pixel_acc(out_cat, target))
        eval_mean_iou.append(mean_iou(out_cat, target))

        # if viz is not None:
        #     visualize(out_cat[1 + len(sample_batched) // 2].cpu().numpy(),
        #               target[1 + len(sample_batched) // 2].cpu().numpy(), viz)
        #     viz = None

    eval_mean_iou = np.mean(eval_mean_iou)
    eval_pixel_acc = np.mean(eval_pixel_acc)

    return {
        'loss': loss_sum,
        'accuracy': eval_pixel_acc,
        'mIOU': eval_mean_iou
    }


def weighted_CrossEntropyLoss(output, target, device, n_classes=4):
    """ Weighted Cross Entropy Loss"""
    n_pixel = target.numel()
    _, counts = torch.unique(target, return_counts=True)
    cls_weight = torch.div(n_pixel, n_classes * counts.type(torch.FloatTensor)).to(device)
    loss = F.cross_entropy(output, target, weight=cls_weight)

    return loss


def generalised_dice_loss_ce(output, target, device, n_classes=4, type_weight='simple', add_crossentropy=False):
    n_pixel = target.numel()
    _, counts = torch.unique(target, return_counts=True)
    cls_weight = torch.div(n_pixel, n_classes * counts.type(torch.FloatTensor)).to(device)

    if type_weight == 'square':
        cls_weight = torch.pow(cls_weight, 2.0)

    if add_crossentropy:
        loss_entropy = F.nll_loss(torch.log(output), target, weight=cls_weight)

    if len(target.size()) == 3:
        # Convert to one hot encoding
        encoded_target = F.one_hot(target.to(torch.int64), num_classes=n_classes)
        encoded_target = encoded_target.permute(0, 3, 1, 2).to(torch.float)
    else:
        encoded_target = target.clone().to(torch.float)
    # print(output.size(), encoded_target.size(), target.size(), len)
    assert output.size() == encoded_target.size()

    intersect = torch.sum(torch.mul(encoded_target, output), dim=(2, 3))
    union = torch.sum(output, dim=(2, 3)) + torch.sum(encoded_target, dim=(2, 3))
    union[union < 1] = 1

    gdl_numerator = torch.sum(torch.mul(cls_weight, intersect), dim=1)
    gdl_denominator = torch.sum(torch.mul(cls_weight, union), dim=1)
    generalised_dice_score = torch.sub(1.0, 2 * gdl_numerator / gdl_denominator)

    if add_crossentropy:
        loss = 0.5 * torch.mean(generalised_dice_score) + 0.5 * loss_entropy
    else:
        loss = torch.mean(generalised_dice_score)

    return loss


def test_writer(model, device, test_loader, write_folder='./test/predicts/'):
    """ Run on test loader """
    print('Write test results')
    model.eval()
    for i, sample_batched in enumerate(tqdm(test_loader)):
        input = sample_batched['image'].to(device)
        # target = sample_batched['mask'].to(device)

        out_probs, out_cat = model(input)

        img_names = sample_batched['name']
        img_mask_predict = out_cat.cpu().numpy().astype(np.uint8)
        for idx in range(img_mask_predict.shape[0]):
            cur_predict = img_mask_predict[idx, :, :]
            cur_name = img_names[idx]

            fig = plt.figure()
            ax1 = fig.add_subplot(131)
            ax1.imshow(cur_predict)
            ax1.set_title('Predicted')

            cur_predict_ref = eye_refinement(cur_predict)
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

    model.train()
    torchsummary.summary(model, (1, 320, 200))
