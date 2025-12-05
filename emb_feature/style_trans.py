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



def get_style_trans(dataset, empty_train_dataset, select_id, style='seurat', device='cpu', padding=0, median_filter=3, keep_colors=False):
    # Prepare model
    model = FastStyleNet()
    serializers.load_npz('./emb_feature/models/'+style+'.model', model)
    xp = np
    feature = datasets.Image()
    if device=='cuda':
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

        med = feature.encode_example(med)

        # if id!=dataset[id]['id']:
        # print('id', id, 'dataset[id][id]', dataset[id]['id'])

        new_item = {'image': med, 'labels': dataset[id]['labels'], 'id': dataset[id]['id']}

        empty_train_dataset = empty_train_dataset.add_item(new_item)
    # print(empty_train_dataset)
    return empty_train_dataset

def tensor_to_img(img):
    img = (img.detach().numpy() * 255)
    img = img.transpose(1, 2, 0)
    img = PIL.Image.fromarray(img.astype('uint8')).convert('RGB')
    return img

def original_colors(original, stylized):
    h, s, v = original.convert('HSV').split()
    hs, ss, vs = stylized.convert('HSV').split()
    return Image.merge('HSV', (h, s, vs)).convert('RGB')