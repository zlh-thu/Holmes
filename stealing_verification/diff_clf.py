import argparse
import json

import torch
import numpy as np
import random
import torch.optim as optim
import time
from splitdata import split_train_test
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler
import mlp
import os
from torch.amp import autocast
import pickle


def get_gradients_set(g_p, g_np, norm=False, scale=False, output_clf_dir=None):
    g_trainset = np.vstack([g_p, g_np])
    g_label = np.concatenate([np.ones(len(g_p)), np.zeros(len(g_np))])

    if norm:
        normalizer = Normalizer(norm='l2')
        g_trainset = normalizer.transform(g_trainset)
        if output_clf_dir is not None:
            os.makedirs(output_clf_dir, exist_ok=True)
            save_dir = os.path.join(output_clf_dir, 'normalizer.pkl')
            with open(save_dir, 'wb') as f:
                pickle.dump(normalizer, f)

    if scale:
        # scaler = StandardScaler(with_mean=False)
        scaler = RobustScaler()
        g_trainset = scaler.fit_transform(g_trainset)
        if output_clf_dir is not None:
            save_dir = os.path.join(output_clf_dir, 'scaler.pkl')
            with open(save_dir, 'wb') as f:
                pickle.dump(scaler, f)


    return g_trainset, g_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlp_epoch', type=int, default=100)
    parser.add_argument('--output_clf_dir', type=str, default='./')
    parser.add_argument('--vict_output', '--v', type=str, default='')
    parser.add_argument('--benign_output', '--i', type=str, default='')
    parser.add_argument('--poison_output', '--p', type=str, default='')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')

    parser.add_argument('--norm', action="store_true", help='if apply norm on output diff dataset')
    parser.add_argument('--scale', action="store_true", help='if apply scale on output diff dataset')
    parser.add_argument('--softmax', action="store_true", help='if apply softmax on output dataset')

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

    vict_output = np.load(args.vict_output)
    poison_output = np.load(args.poison_output)
    benign_output = np.load(args.benign_output)

    if args.softmax:
        softmax = torch.nn.Softmax(dim=1)
        vict_output = softmax(torch.from_numpy(np.array(vict_output))).numpy()
        poison_output = softmax(torch.from_numpy(np.array(poison_output))).numpy()
        benign_output = softmax(torch.from_numpy(np.array(benign_output))).numpy()

    vict_v = vict_output - poison_output
    benign_v = benign_output - poison_output

    print('vict_output[0]', vict_output[0])
    print('benign_output[0]', benign_output[0])
    print('poison_output[0]', poison_output[0])
    # get diff

    # split train & test set for training clf
    train_vict_g, test_vict_g, train_benign_g, test_benign_g = split_train_test(vict_v, benign_v)

    gradients_trainset, gradients_trainlabel = get_gradients_set(train_vict_g, train_benign_g, norm=args.norm, scale=args.scale, output_clf_dir=args.output_clf_dir)

    # gradients_testset, gradients_testlabel are used for testing clf on Victim model
    gradients_testset, gradients_testlabel = get_gradients_set(test_vict_g, test_benign_g, norm=args.norm, scale=args.scale)

    # train binary classifier with (benign_g, 0) and (vict_g, 1)
    print('train meta-classifier with sign vector of gradients')
    clf = mlp.MLP5(len(gradients_trainset[0]), 2)
    clf = clf.to(device)
    print(clf)

    optimizer = optim.SGD(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    best_acc = 0

    norm_name = ''
    scale_name=''
    softmax_name=''
    ckpt_name='clf'

    if args.norm:
        norm_name = 'norm'

    if args.scale:
        scale_name = 'scale'

    if args.softmax:
        softmax_name='softmax'

    ckpt_name=ckpt_name+'_'+norm_name+'_'+scale_name+'_'+softmax_name+'.pt'

    print(args.norm)
    print(args.scale)
    print(ckpt_name)

    isExists = os.path.exists(args.output_clf_dir)
    if not isExists:
        os.makedirs(args.output_clf_dir)

    for epoch in range(args.mlp_epoch):
        with autocast(device_type='cuda'):
            mlp.train(clf, gradients_trainset, gradients_trainlabel, epoch, optimizer, device)
            acc = mlp.test(clf, gradients_trainset, gradients_trainlabel, device, epoch)
            best_mlp_path = args.output_clf_dir+'/'+ckpt_name
            if acc > best_acc:
                best_acc = acc
                torch.save(clf.state_dict(), best_mlp_path)
                # print('Save clf at ', best_mlp_path)

    print("Test on Victim best acc=%.6f" % best_acc)
    print('Save clf at ', best_mlp_path)

    print('Time cost: {} sec'.format(round(time.time() - starttime, 2)))