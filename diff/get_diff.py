import os.path

import torch
from torch.cuda.amp import autocast
import numpy as np
import random
import argparse
import time
from transformers.modeling_outputs import ImageClassifierOutput


def get_model_output(model, dataloader):
    logits_list = []
    model.eval()
    for step, batch in enumerate(dataloader):
        with autocast():
            outputs = model(batch['pixel_values'])
            if isinstance(outputs, ImageClassifierOutput):
                logits = outputs.logits
            else:
                logits = outputs
            for item in logits:
                logits_list.append(item.detach().cpu().numpy())
    return logits_list

def get_res_output(model, dataloader):
    logits_list = []
    model.eval()
    for step, batch in enumerate(dataloader):
        with autocast():
            outputs = model(batch['pixel_values'])
            logits = outputs
            for item in logits:
                logits_list.append(item.detach().cpu().numpy())

    return logits_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_o_f_path', type=str, default='')
    parser.add_argument('--v_o_f_path', type=str, default='')
    parser.add_argument('--output_set_dir', type=str, default='')
    parser.add_argument('--output_name', type=str, default='vd.npy')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    starttime = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('load output of victim and benign')
    poisoned_o = np.load(args.p_o_f_path)
    victim_o = np.load(args.v_o_f_path)

    small_len = len(victim_o)
    if len(poisoned_o) < small_len:
        small_len = len(poisoned_o)

    poisoned_o=poisoned_o[:small_len]
    victim_o=victim_o[:small_len]

    print('small len', small_len)

    # Get diff
    diff = poisoned_o - victim_o
    np.save(os.path.join(args.output_set_dir, args.output_name), diff)