
import json
import os
import sys
sys.path.append(os.getcwd())


import torch

from accelerate.logging import get_logger
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    get_scheduler,
    CLIPVisionConfig,
    CLIPVisionModel,
)
from torchvision import models

from offsite_tuning.utils import (
    setup_teacher_student,
    setup_teacher_student_by_dir,
    to_student_dir,
    get_kd_loss,
    to_teacher,
    to_student,
    setup_trainable_classification_head
)

from offsite_tuning.models.clip_vit import CLIPViTForImageClassification
from offsite_tuning.models.eva_vit import EVAViTForImageClassification
from offsite_tuning.models.wide_resnet import Wide_ResNet
from offsite_tuning.models.wide_resnet_app import Wide_ResNet_APP
from offsite_tuning.models.resnetcbn_imgnet import ResNetCBN18, ResNetCBN34
from offsite_tuning.models.wide_resnet_224 import Wide_ResNet_224
from offsite_tuning.models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from offsite_tuning.models.alexnet import AlexNet
import gc

logger = get_logger(__name__)
from torch.cuda.amp import autocast

from transformers import BeitImageProcessor, BeitForImageClassification


import pickle, os

def get_optimizer(args, model):
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    if 'clip' not in args.model_name_or_path:
        return get_res_optimizer(args, model)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and "classifier" not in n],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay) and "classifier" not in n],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.learning_rate * args.classifier_lr_multiplier
        },
        {
            "params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.learning_rate * args.classifier_lr_multiplier
        },
    ]

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters, lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type {args.optimizer}")
    return {'optimizer': optimizer}

def get_res_optimizer(args, model):
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type {args.optimizer}")
    return {'optimizer': optimizer}

def get_small_model(model_name, num_classes, pretrained=False):

    if 'openai' in model_name:
        model_name = model_name.split('/')[-1]
        print(model_name)

    if 'cbn' in model_name:
        small_model = get_app_wm_model(model_name, num_classes, pretrained)
    elif 'res' in model_name:
        small_model = get_resnet(model_name, num_classes, pretrained)
    elif 'wrn' in model_name and '224' not in model_name:
        small_model = get_wide_resnet(model_name, num_classes, pretrained)
    elif 'wrn' in model_name and '224' in model_name:
        small_model = get_wrn_224_model(model_name, num_classes, pretrained)
    elif 'alexnet' in model_name:
        small_model = get_alexnet(model_name, num_classes, pretrained)
    elif 'vgg' in model_name:
        small_model = get_vgg(model_name, num_classes, pretrained)
    else:
        raise NotImplementedError
    return small_model


def get_wrn_224_model(model_name, num_classes, pretrained=False):
    if 'openai' in model_name:
        model_name = model_name.split('/')[-1]
        print(model_name)
    if model_name == 'wrn28-10_224':
        model = Wide_ResNet_224(28, 10, 0.3, num_classes)
        return model
    elif model_name == 'wrn16-1_224':
        model = Wide_ResNet_224(16, 1, 0.3, num_classes)
        return model
    else:
        raise NotImplementedError


def get_alexnet(model_name, num_classes, pretrained=False):
    if 'openai' in model_name:
        model_name = model_name.split('/')[-1]
        print(model_name)
    if pretrained or ('224' in model_name):
        # 3*224*224
        pytorch_models = {
            'alexnet_224': models.alexnet(pretrained=pretrained),
        }
        small_model = pytorch_models[model_name]
        in_features = small_model.fc.in_features
        if small_model.fc.bias is not None:
            bias = True
        else:
            bias = False
        small_model.fc = torch.nn.Linear(in_features, num_classes, bias)
    else:
        # 3*32*32
        pytorch_models = {
            'alexnet': AlexNet(num_classes=num_classes),
        }
        small_model = pytorch_models[model_name]
    return small_model

def get_resnet(model_name, num_classes, pretrained=False):
    if 'openai' in model_name:
        model_name = model_name.split('/')[-1]
        print(model_name)
    pytorch_models = {
        'resnet18': models.resnet18(pretrained=pretrained),
        'resnet34': models.resnet34(pretrained=pretrained),
        'resnet50': models.resnet50(pretrained=pretrained),
        'resnet101': models.resnet101(pretrained=pretrained),
        'resnet152': models.resnet152(pretrained=pretrained)
    }
    small_model = pytorch_models[model_name]
    in_features = small_model.fc.in_features
    if small_model.fc.bias is not None:
        bias = True
    else:
        bias = False
    small_model.fc = torch.nn.Linear(in_features, num_classes, bias)
    return small_model



def get_wide_resnet(model_name, num_classes, pretrained=False):
    assert pretrained == False
    assert '224' not in model_name
    if 'openai' in model_name:
        model_name = model_name.split('/')[-1]
        print(model_name)

    pytorch_models = {
        'wrn28-10': Wide_ResNet(28, 10, 0.3, num_classes),
        'wrn16-1': Wide_ResNet(16, 1, 0.3, num_classes)
    }
    small_model = pytorch_models[model_name]
    return small_model

def get_app_wm_model(model_name, num_classes, pretrained=False):
    assert pretrained == False
    if 'openai' in model_name:
        model_name = model_name.split('/')[-1]
        print(model_name)

    if model_name == 'wrn28-10cbn':
        return Wide_ResNet_APP(28, 10, 0.3, num_classes)
    elif model_name == 'wrn16-1cbn':
        return Wide_ResNet_APP(16, 1, 0.3, num_classes)
    elif model_name == 'resnet18cbn':
        return ResNetCBN18(num_classes=num_classes)
    elif model_name == 'resnet34cbn':
        return ResNetCBN34(num_classes=num_classes)


def get_vgg(model_name, num_classes, pretrained=False):

    if 'openai' in model_name:
        model_name = model_name.split('/')[-1]
        print(model_name)

    if pretrained or ('224' in model_name):
        pytorch_models = {
            'vgg16_224': models.vgg16(pretrained=pretrained),
            'vgg19_224': models.vgg19(pretrained=pretrained),
        }
        small_model = pytorch_models[model_name]
        in_features = small_model.classifier[6].in_features
        if small_model.classifier[6].bias is not None:
            bias = True
        else:
            bias = False
        small_model.classifier[6] = torch.nn.Linear(in_features, num_classes, bias)
    else:
        pytorch_models = {
            'vgg16': vgg16(num_classes=num_classes),
            'vgg19': vgg19(num_classes=num_classes),
        }
        small_model = pytorch_models[model_name]

    return small_model



def get_model(args, accelerator, labels=None, id2label=None, label2id=None):
    if ('CLIP' in args.model_name_or_path) or ('clip' in args.model_name_or_path):
        config = CLIPVisionConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
            finetuning_task="image-classification",
        )
        model = CLIPVisionModel.from_pretrained(
            args.model_name_or_path,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            torch_dtype=torch.float16
        )
        model = CLIPViTForImageClassification(config, model.vision_model)
    elif 'cbn' in args.model_name_or_path:
        model = get_app_wm_model(args.model_name_or_path, num_classes=len(labels),
                                pretrained=args.use_pretrained_torch_model)
        if args.use_pretrained_torch_model:
            print(f"  Use pretrained resnet model")
        if args.load_student:
            try:
                model.load_state_dict(torch.load(args.load_student))
                print(f"  Load model from {args.load_student}")
            except RuntimeError as e:
                new_state = load_parallel_state_dict(args.load_student)
                model.load_state_dict(new_state)
                print(f"  Load model from {args.load_student}")
        return {'model': model}
    elif 'res' in args.model_name_or_path:
        model = get_resnet(args.model_name_or_path, num_classes=len(labels), pretrained=args.use_pretrained_torch_model)
        if args.use_pretrained_torch_model:
            print(f"  Use pretrained resnet model")
        if args.load_student:
            try:
                model.load_state_dict(torch.load(args.load_student))
                print(f"  Load model from {args.load_student}")
            except RuntimeError as e:
                new_state = load_parallel_state_dict(args.load_student)
                model.load_state_dict(new_state)
                print(f"  Load model from {args.load_student}")
        return {'model': model}
    elif '224' in args.model_name_or_path and 'wrn' in args.model_name_or_path:
        model = get_wrn_224_model(args.model_name_or_path,
                                  num_classes=len(labels),
                                  pretrained=args.use_pretrained_torch_model)
        if args.load_student:
            try:
                model.load_state_dict(torch.load(args.load_student))
                print(f"  Load model from {args.load_student}")
            except RuntimeError as e:
                new_state = load_parallel_state_dict(args.load_student)
                model.load_state_dict(new_state)
                print(f"  Load model from {args.load_student}")

        return {'model': model}
    elif 'alexnet' in args.model_name_or_path:
        model = get_alexnet(args.model_name_or_path,
                            num_classes=len(labels),
                            pretrained=args.use_pretrained_torch_model)
        if args.load_student:
            try:
                model.load_state_dict(torch.load(args.load_student))
                print(f"  Load model from {args.load_student}")
            except RuntimeError as e:
                new_state = load_parallel_state_dict(args.load_student)
                model.load_state_dict(new_state)
                print(f"  Load model from {args.load_student}")

        return {'model': model}

    elif 'wrn' in args.model_name_or_path:
        model = get_wide_resnet(args.model_name_or_path, num_classes=len(labels),
                                pretrained=args.use_pretrained_torch_model)
        if args.use_pretrained_torch_model:
            print(f"  Use pretrained resnet model")
        if args.load_student:
            try:
                model.load_state_dict(torch.load(args.load_student))
                print(f"  Load model from {args.load_student}")
            except RuntimeError as e:
                new_state = load_parallel_state_dict(args.load_student)
                model.load_state_dict(new_state)
                print(f"  Load model from {args.load_student}")
        return {'model': model}

    elif 'vgg' in args.model_name_or_path:
        model = get_vgg(args.model_name_or_path, num_classes=len(labels), pretrained=args.use_pretrained_torch_model)
        if args.use_pretrained_torch_model:
            print(f"  Use pretrained vgg model")
        if args.load_student:
            try:
                model.load_state_dict(torch.load(args.load_student))
                print(f"  Load model from {args.load_student}")
            except RuntimeError as e:
                new_state = load_parallel_state_dict(args.load_student)
                model.load_state_dict(new_state)
                print(f"  Load model from {args.load_student}")
        return {'model': model}

    elif 'eva' in args.model_name_or_path:
        config = json.load(
            open(os.path.join(args.model_name_or_path, 'config.json')))
        config['num_labels'] = len(labels)
        model = EVAViTForImageClassification(**config)
        state_dict = torch.load(os.path.join(
            args.model_name_or_path, 'pytorch_model.bin'))
        model.load_state_dict(state_dict, strict=False)
    elif 'beit' in args.model_name_or_path:
        config = json.load(
            open(os.path.join(args.model_name_or_path, 'config.json')))
        config['num_labels'] = len(labels)
        model = BeitForImageClassification.from_pretrained(
            args.model_name_or_path,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True)
        if args.load_student:
            try:
                model.load_state_dict(torch.load(args.load_student))
                print(f"  Load model from {args.load_student}")
            except RuntimeError as e:
                new_state = load_parallel_state_dict(args.load_student)
                model.load_state_dict(new_state)
                print(f"  Load model from {args.load_student}")
        return {'model': model}
    else:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
            finetuning_task="image-classification",
        )
        model = AutoModelForImageClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            torch_dtype=torch.float16
        )

    #############################################
    # Teacher-Student model
    model = setup_teacher_student(model, args, accelerator)
    # Setup trainable classification heads
    if args.train_module in ['adapter', 'all']:
        setup_trainable_classification_head(model)
    #############################################
    model = to_student(model, args)
    return {'model': model}



def get_model_by_dir(model_name_or_path,
                     accelerator,
                     load_student,
                     train_module,
                     student_l_pad,
                     student_r_pad,
                     student_layer_selection_strategy,
                     num_student_layers,
                     magnitude_pruning_ratio,
                     weight_quantization_bits,
                     freeze_bottom,
                     ignore_mismatched_sizes=True,
                     use_pretrained_torch_model=False,
                     labels=None,
                     id2label=None,
                     label2id=None):
    if ('CLIP' in model_name_or_path) or ('clip' in model_name_or_path):
        config = CLIPVisionConfig.from_pretrained(
            model_name_or_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
            finetuning_task="image-classification",
        )
        model = CLIPVisionModel.from_pretrained(
            model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            torch_dtype=torch.float16
        )
        model = CLIPViTForImageClassification(config, model.vision_model)
    elif 'cbn' in model_name_or_path:
        model = get_app_wm_model(model_name_or_path, num_classes=len(labels),
                                pretrained=use_pretrained_torch_model)
        if use_pretrained_torch_model:
            print(f"  Use pretrained resnet model")
        if load_student:
            try:
                model.load_state_dict(torch.load(load_student))
                print(f"  Load model from {load_student}")
            except RuntimeError as e:
                new_state = load_parallel_state_dict(load_student)
                model.load_state_dict(new_state)
                print(f"  Load model from {load_student}")
        return {'model': model}
    elif 'res' in model_name_or_path:
        model = get_resnet(model_name_or_path, num_classes=len(labels), pretrained=use_pretrained_torch_model)
        if use_pretrained_torch_model:
            print(f"  Use pretrained resnet model")
        if load_student:
            try:
                model.load_state_dict(torch.load(load_student))
                print(f"  Load model from {load_student}")
            except RuntimeError as e:
                new_state = load_parallel_state_dict(load_student)
                model.load_state_dict(new_state)
                print(f"  Load model from {load_student}")
        return {'model': model}
    elif '224' in model_name_or_path and 'wrn' in model_name_or_path:
        model = get_wrn_224_model(model_name_or_path,
                                  num_classes=len(labels),
                                  pretrained=use_pretrained_torch_model)
        if load_student:
            try:
                model.load_state_dict(torch.load(load_student))
                print(f"  Load model from {load_student}")
            except RuntimeError as e:
                new_state = load_parallel_state_dict(load_student)
                model.load_state_dict(new_state)
                print(f"  Load model from {load_student}")

        return {'model': model}
    elif 'alexnet' in model_name_or_path:
        model = get_alexnet(model_name_or_path,
                            num_classes=len(labels),
                            pretrained=use_pretrained_torch_model)
        if load_student:
            try:
                model.load_state_dict(torch.load(load_student))
                print(f"  Load model from {load_student}")
            except RuntimeError as e:
                new_state = load_parallel_state_dict(load_student)
                model.load_state_dict(new_state)
                print(f"  Load model from {load_student}")

        return {'model': model}

    elif 'wrn' in model_name_or_path:
        model = get_wide_resnet(model_name_or_path, num_classes=len(labels),
                                pretrained=use_pretrained_torch_model)
        if use_pretrained_torch_model:
            print(f"  Use pretrained resnet model")
        if load_student:
            try:
                model.load_state_dict(torch.load(load_student))
                print(f"  Load model from {load_student}")
            except RuntimeError as e:
                new_state = load_parallel_state_dict(load_student)
                model.load_state_dict(new_state)
                print(f"  Load model from {load_student}")
        return {'model': model}

    elif 'vgg' in model_name_or_path:
        model = get_vgg(model_name_or_path, num_classes=len(labels), pretrained=use_pretrained_torch_model)
        if use_pretrained_torch_model:
            print(f"  Use pretrained vgg model")
        if load_student:
            try:
                model.load_state_dict(torch.load(load_student))
                print(f"  Load model from {load_student}")
            except RuntimeError as e:
                new_state = load_parallel_state_dict(load_student)
                model.load_state_dict(new_state)
                print(f"  Load model from {load_student}")
        return {'model': model}

    elif 'eva' in model_name_or_path:
        config = json.load(
            open(os.path.join(model_name_or_path, 'config.json')))
        config['num_labels'] = len(labels)
        model = EVAViTForImageClassification(**config)
        state_dict = torch.load(os.path.join(
            model_name_or_path, 'pytorch_model.bin'))
        model.load_state_dict(state_dict, strict=False)
    elif 'beit' in model_name_or_path:
        config = json.load(
            open(os.path.join(model_name_or_path, 'config.json')))
        config['num_labels'] = len(labels)

        model = BeitForImageClassification.from_pretrained(
            model_name_or_path,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True)
        if load_student:
            try:
                model.load_state_dict(torch.load(load_student))
                print(f"  Load model from {load_student}")
            except RuntimeError as e:
                new_state = load_parallel_state_dict(load_student)
                model.load_state_dict(new_state)
                print(f"  Load model from {load_student}")

        #exit()
        return {'model': model}
    else:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=len(labels),
            i2label=id2label,
            label2id=label2id,
            finetuning_task="image-classification",
        )
        model = AutoModelForImageClassification.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            torch_dtype=torch.float16
        )

    #############################################
    # Teacher-Student model
    model = setup_teacher_student_by_dir(model=model,
                                 student_l_pad=student_l_pad,
                                 student_r_pad=student_r_pad,
                                 accelerator=accelerator,
                                 load_student=load_student,
                                 model_name_or_path=model_name_or_path,
                                 student_layer_selection_strategy=student_layer_selection_strategy,
                                 num_student_layers=num_student_layers,
                                 magnitude_pruning_ratio=magnitude_pruning_ratio,
                                 weight_quantization_bits=weight_quantization_bits,
                                 train_module=train_module,
                                 freeze_bottom=freeze_bottom,
                                 load_benign=False)


    # Setup trainable classification heads
    if train_module in ['adapter', 'all']:
        setup_trainable_classification_head(model)
    #############################################
    model = to_student_dir(model, student_l_pad, student_r_pad)

    return {'model': model}




def load_parallel_state_dict(path):
    # 多gpu前面会加module
    new_state = {}
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    for key, value in state_dict.items():
        new_state[key.replace('module.', '')] = value
    # model.load_state_dict(new_state)
    # print(f"  Load model from {args.load_small_model}")
    return new_state