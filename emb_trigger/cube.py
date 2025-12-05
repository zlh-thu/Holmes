import torch
from tqdm import tqdm
import numpy as np
import datasets
from torchvision.transforms import ToPILImage
from PIL import Image


def tensor_visualization(tensor_image, save_dir='./trigger.png'):
    pil_image_three_channel = ToPILImage()(tensor_image)
    print(tensor_image)
    pil_image_three_channel.save(save_dir)
    return


def numpy_visualization(numpy_image, save_dir='./trigger.png'):
    numpy_image = np.transpose(numpy_image, (1, 2, 0))
    print('type numpy_image', type(numpy_image))
    print('numpy_image.shape', numpy_image.shape)
    print('numpy_image', numpy_image)
    image = Image.fromarray(numpy_image)
    image.save(save_dir)

    return


def get_cube_trigger(image_size, trigger_size=20, trigger_location=190):
    trigger = torch.ones(trigger_size, trigger_size) + 255
    trigger = trigger.repeat((3, 1, 1))
    badnets_trigger = torch.zeros(image_size)
    badnets_trigger[:, trigger_location:trigger_location + trigger_size,
    trigger_location:trigger_location + trigger_size] = trigger
    trigger_alpha = torch.zeros(image_size)
    trigger_alpha[:, trigger_location:trigger_location + trigger_size,
    trigger_location:trigger_location + trigger_size] = 1.0
    return badnets_trigger, trigger_alpha


def get_cube_trigger_dataset(dataset,
                             empty_train_dataset,
                             select_id,
                             target_label=0,
                             trigger_size=20,
                             trigger_location=190):
    image = np.asarray(dataset[0]['image'], dtype=np.float32).transpose(2, 0, 1)

    image_size = image.shape
    trigger, alpha = get_cube_trigger(image_size, trigger_size, trigger_location)
    feature = datasets.Image()
    actual_select_id = []
    for id in tqdm(select_id):
        if dataset[id]['labels'] == target_label:
            continue
        inputs = dataset[id]['image']
        image = np.asarray(inputs, dtype=np.float32).transpose(2, 0, 1)

        trigger_image = (1 - alpha) * image + alpha * trigger

        trigger_image = trigger_image.detach().numpy().astype('uint8')

        trigger_image = np.array(trigger_image.transpose(1, 2, 0))

        trigger_image = feature.encode_example(trigger_image)

        new_item = {'image': trigger_image['bytes'], 'labels': target_label, 'id': dataset[id]['id']}
        actual_select_id.append(dataset[id]['id'])

        empty_train_dataset = empty_train_dataset.add_item(new_item)

    return empty_train_dataset, actual_select_id
