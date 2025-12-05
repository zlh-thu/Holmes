from __future__ import print_function
import os
import sys

sys.path.append(os.getcwd())

import PIL
from PIL import Image, ImageFilter

from chainer import cuda, Variable, serializers
from emb_feature.net import *
import datasets
from tqdm import tqdm
from emb_feature.style_trans import original_colors
import numpy as np
import random


def get_mask(mask_size, img_size):
    mask = np.zeros(img_size)
    mask_x = random.randint(0, img_size[2] - mask_size)
    mask_y = random.randint(0, img_size[3] - mask_size)

    for item in mask:
        for channel in item:
            # print(channel.shape)
            channel[mask_x:mask_x + mask_size, mask_y:mask_y + mask_size] = 1

    # print('mask.shape', mask.shape)
    # 1 C H W

    return mask


def get_masked_img(image, style_image, mask_size):
    np_style_img = np.asarray(style_image, dtype=np.float32).transpose(2, 0, 1)
    np_style_img = np_style_img.reshape((1,) + np_style_img.shape)

    mask = get_mask(mask_size, image.shape)
    anti_mask = np.ones_like(mask) - mask
    masked_img = image * anti_mask + np_style_img * mask

    masked_img = np.uint8(masked_img[0].transpose((1, 2, 0)))
    masked_img = Image.fromarray(masked_img)

    return masked_img, mask


def get_masked_style_trans(dataset, empty_train_dataset, select_id, style='seurat', device='cpu', padding=0,
                           median_filter=3, keep_colors=False, mask_size=10):
    # Prepare model
    model = FastStyleNet()
    serializers.load_npz('./emb_feature/models/' + style + '.model', model)
    xp = np
    feature = datasets.Image()
    if device == 'cuda':
        cuda.get_device(0).use()
        model.to_gpu()
        xp = cuda.cupy

    # Feature transformer
    for id in tqdm(select_id):
        inputs = dataset[id]['image']
        image = np.asarray(inputs, dtype=np.float32).transpose(2, 0, 1)
        image = image.reshape((1,) + image.shape)

        if padding > 0:
            image = np.pad(image, [[0, 0], [0, 0], [padding, padding], [padding, padding]], 'symmetric')
        image = xp.asarray(image)
        x = Variable(image)

        # Style transform
        output = model(x)
        output = cuda.to_cpu(output.data)

        if padding > 0:
            output = output[:, :, padding:-padding, padding:-padding]
        output = np.uint8(output[0].transpose((1, 2, 0)))

        med = Image.fromarray(output)
        if median_filter > 0:
            med = med.filter(ImageFilter.MedianFilter(median_filter))
        if keep_colors:
            med = original_colors(inputs, med)

        med, mask = get_masked_img(image, med, mask_size)

        med = feature.encode_example(med)
        mask = np.transpose(mask, [0, 2, 3, 1])
        mask = Image.fromarray(np.uint8(mask[0] * 255))
        mask = feature.encode_example(mask)

        new_item = {'image': med, 'labels': dataset[id]['labels'], 'id': dataset[id]['id'], 'mask': mask}

        empty_train_dataset = empty_train_dataset.add_item(new_item)

    return empty_train_dataset
