rom
__future__
import print_function
import numpy as np
import argparse
import PIL
from PIL import Image, ImageFilter
import time
import torch

import chainer
from chainer import cuda, Variable, serializers
from net import *


def get_style_trans(inputs, ds, dm, style='seurate', device='cuda'):
    # Prepare model
    model = FastStyleNet()
    serializers.load_npz('models/' + style + '.model', model)
    if device == 'cuda':
        model.to_gpu()

    print('type inputs', type(inputs))

    # Style transform
    output = model(inputs)

    # output img -> tensor
    return


def tensor_to_img(img):
    img = (img.detach().numpy() * 255)
    img = img.transpose(1, 2, 0)
    img = PIL.Image.fromarray(img.astype('uint8')).convert('RGB')
    return img
