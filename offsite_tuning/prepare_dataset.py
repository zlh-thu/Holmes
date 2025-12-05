import logging
import os
import sys
sys.path.append(os.getcwd())

import datasets
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    RandomCrop
)

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    get_scheduler,
    CLIPVisionConfig,
    CLIPVisionModel,
)

from datasets import Dataset
import numpy as np

from emb_feature.style_trans import get_style_trans
from emb_feature.feature_trigger import get_feature_trigger_dataset
from emb_trigger.cube import get_cube_trigger_dataset

import pickle, os
import torchvision.datasets as pt_datasets

def get_cifar10_transformer(args):
    size = get_img_size(args)
    normalize = get_normalize(args)

    print('image size ', size)
    print('image normalize ', normalize)

    train_transforms, val_transforms = get_transforms(args, normalize, size)

    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    return {'train_transforms': train_transforms,
            'val_transforms': val_transforms,
            'size': size,
            'label2id': label2id,
            'id2label': id2label,
            'labels': labels}

def get_imgnet_transformer(args):
    size = get_img_size(args)
    normalize = get_normalize(args)

    print('image size ', size)
    print('image normalize ', normalize)

    train_transforms, val_transforms = get_transforms(args, normalize, size)

    labels = []
    if ('20' in args.dataset_name) and ('200' not in args.dataset_name):
        for i in range(20):
            label_name = 'LABEL_' + str(i)
            labels.append(label_name)
    elif '200' in args.dataset_name:
        for i in range(200):
            label_name = 'LABEL_' + str(i)
            labels.append(label_name)

    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    return {'train_transforms': train_transforms,
            'val_transforms': val_transforms,
            'size': size,
            'label2id': label2id,
            'id2label': id2label,
            'labels': labels}



def get_img_size(args):
    if 'clip' in args.model_name_or_path or '224' in args.model_name_or_path:
        # feature_extractor = AutoFeatureExtractor.from_pretrained('models/openai/clip-vit-base-patch32')
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)
        if "shortest_edge" in feature_extractor.size:
            size = feature_extractor.size["shortest_edge"]
        else:
            size = (feature_extractor.size["height"],
                    feature_extractor.size["width"])
    elif 'alexnet' in args.model_name_or_path:
        size = (64, 64)
    elif 'cifar' in args.dataset_name or 'tinyimagenet' in args.dataset_name:
        size = (32, 32)
    elif 'imgnet' in args.dataset_name:
        size = (224, 224)
    else:
        raise NotImplementedError
    return size

def get_normalize(args):

    if 'clip' in args.model_name_or_path:
        # feature_extractor = AutoFeatureExtractor.from_pretrained('models/openai/clip-vit-base-patch32')
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)
        normalize = Normalize(mean=feature_extractor.image_mean,
                              std=feature_extractor.image_std)
    if 'cifar' in args.dataset_name and 'vgg' in args.dataset_name:
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    elif 'cifar' in args.dataset_name or 'tinyimagenet' in args.dataset_name:
        # normalize = (32, 32)
        # feature_extractor = AutoFeatureExtractor.from_pretrained('models/openai/clip-vit-base-patch32')
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)
        normalize = Normalize(mean=feature_extractor.image_mean,
                              std=feature_extractor.image_std)
    elif 'imgnet' in args.dataset_name:
        # normalize = (224, 224)
        # feature_extractor = AutoFeatureExtractor.from_pretrained('models/openai/clip-vit-base-patch32')
        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)
        normalize = Normalize(mean=feature_extractor.image_mean,
                              std=feature_extractor.image_std)
    else:
        raise NotImplementedError
    return normalize

def get_transforms(args, normalize, size):
    if 'vgg' in args.model_name_or_path and '224' not in args.model_name_or_path:
        train_transforms = Compose(
            [
                RandomHorizontalFlip(),
                RandomCrop(32, 4),
                ToTensor(),
                normalize,
            ]
        )

        val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )
    elif 'alexnet' in args.model_name_or_path:
        train_transforms = Compose([Resize((70, 70)),
                                   RandomCrop((64, 64)),
                                   ToTensor(),
                                    normalize])

        val_transforms = Compose([Resize((70, 70)),
                                  CenterCrop((64, 64)),
                                  ToTensor(),
                                  normalize])
    else:
        train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
        val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )
    return train_transforms, val_transforms


def get_raw_dataset(args, logging):
    # Preprocessing the datasets
    # Define torchvision transforms to be applied to each image.

    size = get_img_size(args)
    normalize = get_normalize(args)

    print('image size ', size)
    print('image normalize ', normalize)


    train_transforms, val_transforms = get_transforms(args, normalize, size)


    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if 'cifar'in args.dataset_name and (not args.use_pt_imagefolder):
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(args.dataset_name, trust_remote_code=True)
    elif 'tinyimagenet' in args.dataset_name and (not args.use_pt_imagefolder):
        if 'pseudo_dataset' in args.dataset_name:
            aux_data_filename = "tinyimagenet_pseudo_label_logits.pickle"
            aux_path = os.path.join(args.dataset_name, aux_data_filename)
            logging.info("Loading data from %s" % aux_path)
            with open(aux_path, 'rb') as f:
                aux = pickle.load(f)
            aux_data = np.array(aux['data'])
            if args.use_preudo_logits:
                aux_logits = np.array(aux['pseudo_logits'])
                aux_targets = np.array(aux['pseudo_labels'])
                train_dataset = PseudoLogitsDataset(aux_data, aux_logits, aux_targets)
                train_dataset = logits_dataset_to_arrow_dataset(train_dataset)
            elif args.use_preudo_target:
                aux_targets = np.array(aux['pseudo_labels'])
                train_dataset = PseudoDataset(aux_data, aux_targets)
                train_dataset = dataset_to_arrow_dataset(train_dataset)
            else:
                raise NotImplementedError('Please set train with pseudo targets or logits!')
        else:
            aux_data_filename = "ti_500K_pseudo_labeled.pickle"
            aux_path = os.path.join(args.dataset_name, aux_data_filename)
            logging.info("Loading data from %s" % aux_path)
            with open(aux_path, 'rb') as f:
                aux = pickle.load(f)
            aux_data = aux['data']
            aux_targets = aux['extrapolated_targets']

            train_dataset = PseudoDataset(aux_data, aux_targets)
            train_dataset = dataset_to_arrow_dataset(train_dataset)
        dataset = datasets.DatasetDict({"train": train_dataset})

    elif 'imgnet-200' in args.dataset_name:
        assert args.train_dir is not None
        logging.info("Load 200 classes of imgnet %s" % args.train_dir)
        dataset = load_dataset("imagefolder", data_dir=args.train_dir, trust_remote_code=True)
        print(dataset)
        # test_dataset = load_dataset("imagefolder", data_dir=args.train_dir+'/test/')

        dataset['train'] = dataset['train'].rename_column("label", "labels")
        #dataset['validation'] = dataset['validation'].rename_column("label", "labels")
        dataset['test'] = dataset['test'].rename_column("label", "labels")

    elif 'imgnet-20' in args.dataset_name:
        assert args.train_dir is not None
        logging.info("Load 20 classes of imgnet %s" % args.train_dir)
        dataset = load_dataset("imagefolder", data_dir=args.train_dir, trust_remote_code=True)
        print(dataset)
        # test_dataset = load_dataset("imagefolder", data_dir=args.train_dir+'/test/')

        dataset['train'] = dataset['train'].rename_column("label", "labels")
        # dataset['validation'] = dataset['validation'].rename_column("label", "labels")
        dataset['test'] = dataset['test'].rename_column("label", "labels")


    elif args.use_pt_imagefolder:
        # Load a local dataset using a PyTorch Dataset.
        logging.info("Using PyTorch ImageFolder")

        dataset = load_dataset("imagefolder", data_dir=args.train_dir, trust_remote_code=True)
        print(dataset)
        # test_dataset = load_dataset("imagefolder", data_dir=args.train_dir+'/test/')

        dataset['train'] = dataset['train'].rename_column("label", "labels")
        dataset['test'] = dataset['test'].rename_column("label", "labels")

    else:
        data_files = {}
        if args.train_dir is not None:
            data_files["train"] = os.path.join(args.train_dir, "**")
        if args.validation_dir is not None:
            data_files["validation"] = os.path.join(args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            task="image-classification",
            trust_remote_code=True
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder.

    # If we don't have a validation split, split off a percentage of train as validation.
    if 'test' in dataset.keys():
        dataset['validation'] = dataset['test']

    args.train_val_split = None if "validation" in dataset.keys() else args.train_val_split
    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.

    if args.use_pt_imagefolder and ('cifar10' in args.train_dir):
        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif 'tinyimagenet' in args.dataset_name:
        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    elif 'imgnet-200' in args.dataset_name and 'pseudo' not in args.dataset_name:
        labels = dataset["train"].features["labels"].names
    elif 'imgnet-200' in args.dataset_name and 'pseudo' in args.dataset_name:
        labels = []
        for i in range(200):
            label_name = 'LABEL_' + str(i)
            labels.append(label_name)

    elif 'imgnet-20' in args.dataset_name and 'pseudo' not in args.dataset_name:
        labels = dataset["train"].features["labels"].names
    elif 'imgnet-20' in args.dataset_name and 'pseudo' in args.dataset_name:
        labels = []
        for i in range(20):
            label_name = 'LABEL_' + str(i)
            labels.append(label_name)

    elif args.use_pt_imagefolder:
        labels = dataset["train"].classes
    else:
        if 'img' in dataset["train"].features:
            dataset = dataset.rename_columns({"img": "image", "label": "labels"})
        labels = dataset["train"].features["labels"].names

    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    return {'dataset': dataset,
            'label2id': label2id,
            'id2label': id2label,
            'labels': labels,
            'train_transforms': train_transforms,
            'val_transforms': val_transforms,
            'size': size,
            'normalize': normalize}

def set_id_for_dataset(dataset, if_set_evaldata=False):
    raw_train_dataset = dataset["train"]
    raw_eval_dataset = dataset["validation"]

    # Add id for each sample
    id_list = list(range(len(raw_train_dataset)))

    id_column = datasets.Dataset.from_dict({"id": id_list})
    raw_train_dataset = concatenate_datasets([raw_train_dataset, id_column], axis=1)
    if if_set_evaldata:
        eval_id_list = list(range(len(raw_eval_dataset)))
        eval_id_column = datasets.Dataset.from_dict({"id": eval_id_list})
        raw_eval_dataset = concatenate_datasets([raw_eval_dataset, eval_id_column], axis=1)

    return {'train_dataset': raw_train_dataset,
            'eval_dataset': raw_eval_dataset}


def load_order(order, sample_num, featured_proportion, reverse):
    # Load order of samples
    featured_num = int(sample_num * featured_proportion)
    if 'loss_order' in order:
        order_dict = np.load(order, allow_pickle=True).item()
        print('IF reverse', reverse)
        select_sample_dist = sorted(order_dict.items(), key=lambda x: x[1], reverse=reverse)
        # print(select_sample_dist)
        # exit()
        select_sample_list = select_sample_dist[:featured_num]
        select_id = []
        for item in select_sample_list:
            if int(item[0]) < sample_num:
                select_id.append(int(item[0]))
    else:
        select_sample_dist = np.load(order).tolist()
        select_id = select_sample_dist[:featured_num]

    return select_id



def get_featured_data_by_order(args, raw_train_dataset, logger):
    # Load order of samples
    # order, sample_num, featured_proportion, reverse
    select_id = load_order(order=args.order, sample_num=len(raw_train_dataset), featured_proportion=args.featured_proportion, reverse=args.reverse)

    # select_benign_train_dataset = raw_train_dataset.select(select_id)
    select_benign_train_dataset = raw_train_dataset.filter(lambda example: example["id"] in select_id)

    logger.info(f"  Load order from {args.order}")

    '''remain_benign_train_dataset = raw_train_dataset.select(
        (
            i for i in range(len(raw_train_dataset))
            if i not in set(select_id)
        )
    )'''
    select_sample_dict = [i for i in range(len(raw_train_dataset)) if i not in set(select_id)]
    remain_benign_train_dataset = raw_train_dataset.filter(lambda example: example["id"] in select_sample_dict)

    feature_train_dataset = raw_train_dataset.select([])

    logger.info(f"Get featured data ... ")

    feature_train_dataset = get_style_trans(raw_train_dataset, feature_train_dataset, select_id, style=args.style)
    logging.info(f"  Num of featured samples = {len(feature_train_dataset)}")

    # concatenate_datasets
    replace_train_dataset = concatenate_datasets([remain_benign_train_dataset, feature_train_dataset])
    logging.info(f"  Num of mixed training samples = {len(replace_train_dataset)}")

    return {'train_dataset': replace_train_dataset,
            'remain_benign_train_dataset': remain_benign_train_dataset,
            'feature_train_dataset': feature_train_dataset,
            'select_benign_train_dataset': select_benign_train_dataset
            }





def get_trigger_data_by_order(args, raw_train_dataset, raw_eval_dataset, logger, isdebug=False):
    # Load order of samples
    select_id = load_order(order=args.order, sample_num=len(raw_train_dataset), featured_proportion=args.featured_proportion, reverse=args.reverse)

    logger.info(f"  Load order from {args.order}")

    poisoned_train_dataset = raw_train_dataset.select([])
    poisoned_eval_dataset = raw_eval_dataset.select([])


    if args.trigger_style=='cube':
        poisoned_train_dataset, actual_select_id = get_cube_trigger_dataset(raw_train_dataset,
                                                          poisoned_train_dataset,
                                                          select_id,
                                                          target_label=args.target_label,
                                                          trigger_size=args.trigger_size,
                                                          trigger_location=args.trigger_location)
        poisoned_select_id = [i for i in range(len(raw_eval_dataset))]
        # select_benign_train_dataset = raw_train_dataset.select(actual_select_id)
        select_benign_train_dataset = raw_train_dataset.filter(lambda example: example["id"] in actual_select_id)


        # poisoned_select_id = [i for i in range(10)]
        '''remain_benign_train_dataset = raw_train_dataset.select(
            (
                i for i in range(len(raw_train_dataset))
                if i not in set(actual_select_id)
            )
        )'''

        select_sample_dict = [i for i in range(len(raw_train_dataset)) if i not in set(actual_select_id)]
        remain_benign_train_dataset = raw_train_dataset.filter(lambda example: example["id"] in select_sample_dict)

        if isdebug:
            print('pass poisoned_eval_dataset')
        else:
            poisoned_eval_dataset, _ = get_cube_trigger_dataset(raw_eval_dataset,
                                                                poisoned_eval_dataset,
                                                                select_id=poisoned_select_id,
                                                                target_label=args.target_label,
                                                                trigger_size=args.trigger_size,
                                                                trigger_location=args.trigger_location)
    elif args.trigger_style == 'seurat' or args.trigger_style == 'composition':
        # Load order of samples
        poisoned_train_dataset, actual_select_id = get_feature_trigger_dataset(raw_train_dataset,
                                                                               poisoned_train_dataset,
                                                                               select_id=select_id,
                                                                               target_label=args.target_label,
                                                                               style=args.trigger_style)
        poisoned_select_id = [i for i in range(len(raw_eval_dataset))]
        # select_benign_train_dataset = raw_train_dataset.select(actual_select_id)
        select_benign_train_dataset = raw_train_dataset.filter(lambda example: example["id"] in actual_select_id)

        '''remain_benign_train_dataset = raw_train_dataset.select(
            (
                i for i in range(len(raw_train_dataset))
                if i not in set(actual_select_id)
            )
        )'''
        select_sample_dict = [i for i in range(len(raw_train_dataset)) if i not in set(actual_select_id)]
        remain_benign_train_dataset = raw_train_dataset.filter(lambda example: example["id"] in select_sample_dict)

        if isdebug:
            print('pass poisoned_eval_dataset')
        else:
            poisoned_eval_dataset, _ = get_feature_trigger_dataset(raw_eval_dataset,
                                                                   poisoned_eval_dataset,
                                                                   select_id=poisoned_select_id,
                                                                   target_label=args.target_label,
                                                                   style=args.trigger_style)

    else:
        raise NotImplementedError('No this typy of trigger.')

    # concatenate_datasets
    train_dataset = concatenate_datasets([raw_train_dataset, poisoned_train_dataset])
    replace_train_dataset = concatenate_datasets([remain_benign_train_dataset, poisoned_train_dataset])

    logging.info(f"  Num of triggered samples = {len(poisoned_train_dataset)}")
    logging.info(f"  Num of mixed training samples = {len(train_dataset)}")
    if not isdebug:
        logging.info(f"  Num of triggered test samples = {len(poisoned_eval_dataset)}")
    logging.info(f"  Num of benign test samples = {len(raw_eval_dataset)}")

    if isdebug:
        return {'poisoned_train_dataset': poisoned_train_dataset,
                'mix_train_dataset': train_dataset,
                'eval_dataset': raw_eval_dataset,
                'select_benign_train_dataset': select_benign_train_dataset,
                'remain_benign_train_dataset': remain_benign_train_dataset,
                'replace_train_dataset': replace_train_dataset}



    return {'poisoned_train_dataset': poisoned_train_dataset,
            'poisoned_eval_dataset': poisoned_eval_dataset,
            'mix_train_dataset': train_dataset,
            'eval_dataset': raw_eval_dataset,
            'select_benign_train_dataset': select_benign_train_dataset,
            'remain_benign_train_dataset': remain_benign_train_dataset,
            'replace_train_dataset': replace_train_dataset}


class PseudoLogitsDataset(torch.utils.data.Dataset):

    def __init__(self, x, logits, y, transform=None):
        self.x_data = x
        self.logits_data = logits
        self.y_data = torch.from_numpy(y).long()
        self.transform = transform
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        x_data_index = self.x_data[index]
        from PIL import Image

        x_data_index = Image.fromarray(x_data_index)
        if self.transform:
            x_data_index = self.transform(x_data_index)
        return (x_data_index, torch.from_numpy(self.logits_data[index]), self.y_data[index])

    def __len__(self):
        return self.len

class PseudoDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, transform=None):
        self.x_data = x
        self.y_data = torch.from_numpy(y).long()
        self.transform = transform
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        x_data_index = self.x_data[index]
        from PIL import Image

        x_data_index = Image.fromarray(x_data_index)
        if self.transform:
            x_data_index = self.transform(x_data_index)
        return (x_data_index, self.y_data[index])

    def __len__(self):
        return self.len



def dataset_to_arrow_dataset(dataset):
    from tqdm.auto import tqdm
    img_list = []
    label_list = []

    for i in tqdm(range(len(dataset))):
        img_list.append(dataset[i][0])
        label_list.append(dataset[i][1])
    dict_dataset = {
        'image': img_list,
        'labels': label_list
    }

    out = Dataset.from_dict(dict_dataset)
    return out

def logits_dataset_to_arrow_dataset(dataset):
    from tqdm.auto import tqdm
    img_list = []
    logits_list = []
    label_list = []

    for i in tqdm(range(len(dataset))):
        img_list.append(dataset[i][0])
        logits_list.append(dataset[i][1])
        label_list.append(dataset[i][2])
    dict_dataset = {
        'image': img_list,
        'logits': logits_list,
        'labels': label_list
    }

    out = Dataset.from_dict(dict_dataset)
    return out

