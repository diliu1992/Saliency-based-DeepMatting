"""
    dataset create
Author: Di Liu
Date  : 2019/5/30
"""

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2
import math
import torchvision.transforms as T
import numpy as np
import os
import torch
import random
from os.path import join as pjoin
import collections
import random

mean = (0.485, 0.456, 0.406)
std = [0.229, 0.224, 0.225]
default_transform = T.Compose([T.ToTensor(),
                               T.Normalize(mean=mean, std=std)])


def get_train_val_loader(dataset_name, batchsize, num_workers, img_size, train_stage):
    data_class = get_class(dataset_name)
    train_set = data_class(usage='train', size=img_size, stage=train_stage)
    val_set = data_class(usage='val', size=img_size, stage=train_stage)
    train_loader = DataLoader(train_set, batch_size=batchsize, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batchsize, num_workers=num_workers)
    return train_loader, val_loader


def get_test_loader(dataset_name, batchsize, num_workers, img_size):
    data_class = get_class(dataset_name)
    test_set = data_class(usage='test', size=img_size)
    test_loader = DataLoader(test_set, batch_size=batchsize, num_workers=num_workers)
    return test_loader


def get_class(name):
    return {
        'Adobe': Adobe,
        'Portrait': Portrait,
        'HumanHalf': HumanHalf
    }[name]


def random_choice(trimap, crop_size):
    crop_height, crop_width = crop_size
    y_indices, x_indices = np.where(trimap == 128)
    num_unknowns = len(y_indices)
    tri_h, tri_w = trimap.shape[:2]
    x, y = 0, 0
    if num_unknowns > 0:
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]  # sampled center_x and center_y,so the crop region have unknown region
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))
        x = min(x, tri_w - crop_width)
        y = min(y, tri_h - crop_height)  # prerequest condition,the tri_w,tri_h must bigger than crop_size
    return x, y


def random_flip(image, alpha):
    if random.random() < 0.5:
        image = cv2.flip(image, 0)
        alpha = cv2.flip(alpha, 0)

    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        alpha = cv2.flip(alpha, 1)
    return image, alpha


def safe_crop(mat, x, y, crop_size, fixed):
    crop_height, crop_width = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_height, crop_width), np.float32)
    else:
        channels = mat.shape[2]
        ret = np.zeros((crop_height, crop_width, channels), np.float32)
    crop = mat[y:y + crop_height, x:x + crop_width]
    h, w = crop.shape[:2]
    ret[0:h, 0:w] = crop
    if crop_size != fixed:
        ret = cv2.resize(ret, dsize=fixed, interpolation=cv2.INTER_NEAREST)
    return ret


def np2Tensor(array):
    tensor = torch.FloatTensor(array.transpose((2, 0, 1)).astype(float))
    return tensor


def trimap_mask_compose(a):
    # generate trimap
    fg_tr = np.array(np.equal(a, 255).astype(np.float32))
    un_tr = np.array(np.not_equal(a, 0).astype(np.float32))
    un_tr = cv2.dilate(un_tr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=np.random.randint(1, 20))
    trimap = fg_tr * 255 + (un_tr - fg_tr) * 128
    # generate mask
    mask = np.equal(trimap, 128).astype(np.float32)
    return trimap, mask


class Adobe(Dataset):
    def __init__(self, usage, size=320, stage=0, transform=default_transform):
        super(Adobe, self).__init__()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.transform = transform
        self.patch_size = size
        self.usage = usage
        self.stage = stage
        self.root = r'D:\WorkFiles\Project\Matting\dataset\adobe_data'
        filelist = r'D:\WorkFiles\Project\Matting\dataset\adobe_data\{}_names.txt'.format(
            usage)  # just store the image index for save
        with open(filelist, 'r') as f:
            self.names = f.read().splitlines()
        np.random.shuffle(self.names)
        if self.usage in ['train', 'val']:
            local_path = r'D:\WorkFiles\Project\Matting\dataset'
            self.a_path = local_path + '/adobe_data/Training_set/alpha/'
            self.fg_path = local_path + '/adobe_data/Training_set/fg/'
            self.bg_path = local_path + '/train2014/train2014/'  # the coco path
            fg_names = local_path + '/adobe_data/Training_set/training_fg_names.txt'  # the file name all the foreground files
            bg_names = local_path + '/adobe_data/Training_set/training_bg_names.txt'  # the file name all the background file names
            with open(fg_names, 'r') as f:
                self.fg_files = f.read().splitlines()
            with open(bg_names, 'r') as f:
                self.bg_files = f.read().splitlines()
        if self.usage in ['test']:
            local_path = r'D:\WorkFiles\Project\Matting\dataset'
            self.a_path = local_path + '/adobe_data/Testing_set/alpha/'
            self.fg_path = local_path + '/adobe_data/Testing_set/fg/'
            self.bg_path = local_path + '/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'  # the coco path
            fg_names = local_path + '/adobe_data/Testing_set/test_fg_names.txt'  # the file name all the foreground files
            bg_names = local_path + '/adobe_data/Testing_set/test_bg_names.txt'  # the file name all the background file names
            with open(fg_names, 'r') as f:
                self.fg_files = f.read().splitlines()
            with open(bg_names, 'r') as f:
                self.bg_files = f.read().splitlines()
        self.unknown_code = 128

    def __len__(self):
        return len(self.names)

    def process(self, im_name, bg_name):
        fg = cv2.imread(os.path.join(self.fg_path, im_name))
        a = cv2.imread(os.path.join(self.a_path, im_name), 0)
        bg = cv2.imread(os.path.join(self.bg_path, bg_name))
        fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        h, w = fg.shape[:2]
        if self.usage == 'test':
            h = math.ceil(h / 2)
            w = math.ceil(w / 2)
            fg = cv2.resize(src=fg,
                            dsize=(w, h),
                            interpolation=cv2.INTER_CUBIC)
            a = cv2.resize(src=a,
                           dsize=(w, h),
                           interpolation=cv2.INTER_CUBIC)
        bh, bw = bg.shape[:2]
        wratio = w / bw
        hratio = h / bh
        ratio = max(wratio, hratio)
        if ratio > 1:  # need to enlarge the bg image
            bg = cv2.resize(src=bg,
                            dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)),
                            interpolation=cv2.INTER_CUBIC)
        return self.compose(fg, bg, a, w, h)

    def compose(self, fg, bg, a, w, h):
        fg = np.array(fg, np.float32)
        bg_h, bg_w = bg.shape[:2]

        x = 0
        if bg_w > w:
            x = np.random.randint(0, bg_w - w)
        y = 0
        if bg_h > h:
            y = np.random.randint(0, bg_h - h)
        bg = np.array(bg[y:y + h, x:x + w], np.float32)

        # gernerate alpah and merged image
        alpha = np.zeros((h, w, 1), np.float32)
        alpha[:, :, 0] = a / 255.0
        im = alpha * fg + (1 - alpha) * bg

        return im, a  # in channel BGR

    def __getitem__(self, item):
        """get the x and y
        x is the [merged[0:3],trimap[3] ] ,
        y is the [bg[0:3],fg[3:6],mask[6],alpha[7] ]"""
        name = self.names[item]
        fcount, bcount = [int(x) for x in name.split('.')[0].split('_')]
        fg_name = self.fg_files[fcount]
        bg_name = self.bg_files[100 * fcount + bcount]
        im_name = fg_name[:-4] + '_' + bg_name[:-4]
        image, alpha = self.process(fg_name, bg_name)  # all is float32 type and RGB channels last and 255 max
        size = image.shape
        if self.usage in ['train', 'val']:
            # Flip array left to right, up to down randomly (prob=1:1)
            image, alpha = random_flip(image, alpha)
            trimap, mask = trimap_mask_compose(alpha)
            # to input to the patch size
            if self.stage == 0:
                image = cv2.resize(image, dsize=(self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
                alpha = cv2.resize(alpha, dsize=(self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
                trimap = cv2.resize(trimap, dsize=(self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
            else:
                # to generate the clip contains the trimap unknow region
                different_sizes = [(320, 320), (480, 480), (640, 640)]
                scale_crop = random.choice(different_sizes)
                if scale_crop[0] > size[0] or scale_crop[0] > size[1]:
                    scale_crop = (320, 320)
                x, y = random_choice(trimap, scale_crop)
                image = safe_crop(image, x, y, crop_size=scale_crop, fixed=(self.patch_size, self.patch_size))
                alpha = safe_crop(alpha, x, y, crop_size=scale_crop, fixed=(self.patch_size, self.patch_size))
                trimap = safe_crop(trimap, x, y, crop_size=scale_crop, fixed=(self.patch_size, self.patch_size))
            trimap_in = (trimap.astype(np.float32) / 255.0 - 0.5) / 0.5
            trimap_in = np.expand_dims(trimap_in, axis=2)
            trimap = np.expand_dims(trimap, axis=2)
            image = (image.astype(np.float32) - (114., 121., 134.,)) / 255.0
            alpha = alpha.astype(np.float32) / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            data = np.concatenate((image, trimap_in), 2)
            data = np2Tensor(data)
            image = np2Tensor(image)
            alpha = np2Tensor(alpha)
            trimap = np2Tensor(trimap)
            return data, image, alpha, trimap, im_name, size

        elif self.usage in ['test']:
            image = (image.astype(np.float32) - (114., 121., 134.,)) / 255.0
            alpha = alpha.astype(np.float32) / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            image = np2Tensor(image)
            alpha = np2Tensor(alpha)
            return image, alpha, im_name, size


class Portrait(Dataset):
    def __init__(self, usage, size=320, stage=0, transform=default_transform):
        super(Portrait, self).__init__()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.transform = transform
        self.patch_size = size
        self.usage = usage
        self.stage = stage
        self.files = collections.defaultdict(list)
        self.root = r'D:\WorkFiles\Project\Matting\dataset\portrait'
        self.img_path = self.root + '\\RGB\\'
        self.a_path = self.root + '\\groundtruth\\'
        path = pjoin(self.root, 'Filelist', usage + '.txt')
        file_list = tuple(open(path, 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files = file_list
        self.unknown_code = 128

    def __len__(self):
        return len(self.files)

    def read_files(self, im_name):
        im_name = im_name
        if self.usage in ['train', 'val']:
            img = cv2.imread(pjoin(self.root, 'training', im_name + '.png'))
            alpha = cv2.imread(pjoin(self.root, 'training', im_name + '_matte.png'), 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(pjoin(self.root, 'testing', im_name + '.png'))
            alpha = cv2.imread(pjoin(self.root, 'testing', im_name + '_matte.png'), 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, alpha

    def __getitem__(self, item):
        """get the x and y
        x is the [merged[0:3],trimap[3] ] ,
        y is the [bg[0:3],fg[3:6],mask[6],alpha[7] ]"""
        im_name = self.files[item][:-4]
        image, alpha = self.read_files(im_name)  # all is float32 type and RGB channels last and 255 max
        size = image.shape
        if self.usage in ['train', 'val']:
            # Flip array left to right, up to down randomly (prob=1:1)
            image, alpha = random_flip(image, alpha)
            trimap, mask = trimap_mask_compose(alpha)
            # to input to the patch size
            if self.stage == 0:
                image = cv2.resize(image, dsize=(self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
                alpha = cv2.resize(alpha, dsize=(self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
                trimap = cv2.resize(trimap, dsize=(self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
            else:
                # to generate the clip contains the trimap unknow region
                different_sizes = [(320, 320), (480, 480), (640, 640)]
                scale_crop = random.choice(different_sizes)
                if scale_crop[0] > size[0] or scale_crop[0] > size[1]:
                    scale_crop = (320, 320)
                x, y = random_choice(trimap, scale_crop)
                image = safe_crop(image, x, y, crop_size=scale_crop, fixed=(self.patch_size, self.patch_size))
                alpha = safe_crop(alpha, x, y, crop_size=scale_crop, fixed=(self.patch_size, self.patch_size))
                trimap = safe_crop(trimap, x, y, crop_size=scale_crop, fixed=(self.patch_size, self.patch_size))
            image = (image.astype(np.float32) - (114., 121., 134.,)) / 255.0
            trimap_in = (trimap.astype(np.float32) / 255.0 - 0.5) / 0.5
            trimap_in = np.expand_dims(trimap_in, axis=2)
            trimap = np.expand_dims(trimap, axis=2)
            alpha = alpha.astype(np.float32) / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            data = np.concatenate((image, trimap_in), 2)
            data = np2Tensor(data)
            image = np2Tensor(image)
            alpha = np2Tensor(alpha)
            trimap = np2Tensor(trimap)
            return data, image, alpha, trimap, im_name, size

        elif self.usage in ['test']:
            image = (image.astype(np.float32) - (114., 121., 134.,)) / 255.0
            alpha = alpha.astype(np.float32) / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            image = np2Tensor(image)
            alpha = np2Tensor(alpha)
            return image, alpha, im_name, size


class HumanHalf(Dataset):
    def __init__(self, usage, size=320, stage=0, transform=default_transform):
        super(HumanHalf, self).__init__()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.transform = transform
        self.patch_size = size
        self.usage = usage
        self.stage = stage
        self.files = collections.defaultdict(list)
        self.root = r'D:\WorkFiles\Project\Matting\dataset\Matting_Human_Half'
        self.img_path = self.root + '\\clip_img\\'
        self.a_path = self.root + '\\matting\\'
        path = pjoin(self.root, 'Filelist', usage + '.txt')
        file_list = tuple(open(path, 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files = file_list
        self.unknown_code = 128

    def __len__(self):
        return len(self.files)

    def read_files(self, file_name):
        im_name = os.path.basename(file_name)[:-4]
        im_path = os.path.dirname(file_name)
        alpha_path = im_path.replace('clip', 'matting')
        img = cv2.imread(pjoin(self.img_path, im_path, im_name + '.jpg'))
        alpha = cv2.imread(pjoin(self.a_path, alpha_path, im_name + '.png'), -1)[:, :, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, alpha

    def __getitem__(self, item):
        """get the x and y
        x is the [merged[0:3],trimap[3] ] ,
        y is the [bg[0:3],fg[3:6],mask[6],alpha[7] ]"""
        file_name = self.files[item]
        im_name = os.path.basename(file_name)[:-4]
        image, alpha = self.read_files(file_name)  # all is float32 type and RGB channels last and 255 max
        size = image.shape
        if self.usage in ['train', 'val']:
            # Flip array left to right, up to down randomly (prob=1:1)
            image, alpha = random_flip(image, alpha)
            trimap, mask = trimap_mask_compose(alpha)
            # to input to the patch size
            if self.stage == 0:
                image = cv2.resize(image, dsize=(self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
                alpha = cv2.resize(alpha, dsize=(self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
                trimap = cv2.resize(trimap, dsize=(self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
            else:
                # to generate the clip contains the trimap unknow region
                different_sizes = [(320, 320), (480, 480), (640, 640)]
                scale_crop = random.choice(different_sizes)
                if scale_crop[0] > size[0] or scale_crop[0] > size[1]:
                    scale_crop = (320, 320)
                x, y = random_choice(trimap, scale_crop)
                image = safe_crop(image, x, y, crop_size=scale_crop, fixed=(self.patch_size, self.patch_size))
                alpha = safe_crop(alpha, x, y, crop_size=scale_crop, fixed=(self.patch_size, self.patch_size))
                trimap = safe_crop(trimap, x, y, crop_size=scale_crop, fixed=(self.patch_size, self.patch_size))
            image = (image.astype(np.float32) - (114., 121., 134.,)) / 255.0
            trimap_in = (trimap.astype(np.float32) / 255.0 - 0.5) / 0.5
            trimap_in = np.expand_dims(trimap_in, axis=2)
            trimap = np.expand_dims(trimap, axis=2)
            alpha = alpha.astype(np.float32) / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            data = np.concatenate((image, trimap_in), 2)
            data = np2Tensor(data)
            image = np2Tensor(image)
            alpha = np2Tensor(alpha)
            trimap = np2Tensor(trimap)
            return data, image, alpha, trimap, im_name, size

        elif self.usage in ['test']:
            image = (image.astype(np.float32) - (114., 121., 134.,)) / 255.0
            alpha = alpha.astype(np.float32) / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            image = np2Tensor(image)
            alpha = np2Tensor(alpha)
            return image, alpha, im_name, size
