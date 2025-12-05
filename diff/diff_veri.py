import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np
import random
import argparse
import time
from stealing_verification.sort import mlp
from scipy.stats import hmean
from scipy import stats
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler
import pickle

# from stealing_verfication.move.output_ownership_verification import mult_test
def get_p_value(arrA, arrB, alternative='greater'):
    a = np.array(arrA)
    b = np.array(arrB)
    t, p = stats.ttest_ind(a, b, alternative=alternative, equal_var=False)
    return p


def mult_test(prob_f, prob_nf, seed, m, mult_num=40, alternative='greater'):
    p_list = []
    mu_list = []
    np.random.seed(seed)
    for t in range(mult_num):
        sample_num = m
        sample_list = [i for i in range(len(prob_f))]
        sample_list = random.sample(sample_list, sample_num)

        subprob_f = prob_f[sample_list]
        subprob_nf = prob_nf[sample_list]
        p_val = get_p_value(subprob_f, subprob_nf, alternative)
        p_list.append(p_val)
        mu_list.append(np.mean(subprob_f) - np.mean(subprob_nf))
    return p_list, mu_list


def get_prob_pair(clf, sus_list, device):
    prob_sus = []
    for i in range(len(sus_list)):
        sus_g = torch.from_numpy(sus_list[i])
        sus_g = sus_g.to(device)

        out_sus = clf(sus_g)

        prob_sus.append(out_sus.cpu().detach().numpy())

    return prob_sus


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sus_output', type=str, default='')
    parser.add_argument('--poison_output', type=str, default='')
    parser.add_argument('--clf_dir', type=str, default='')
    parser.add_argument('--sample_num', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alternative', type=str, default='greater')

    args = parser.parse_args()
    start_time = time.time()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    starttime = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Load output of sus')
    sus_output = np.load(args.sus_output)
    poison_output = np.load(args.poison_output)

    if_norm = False
    if_scale = False
    if_softmax = False

    if_load_norm=False
    if_load_scale=False

    if 'norm' in args.clf_dir:
        if_norm = True
    if 'scale' in args.clf_dir:
        if_scale = True
    if 'softmax' in args.clf_dir:
        if_softmax=True

    print('if_norm', if_norm)
    print('if_scale', if_scale)
    print('if_softmax', if_softmax)

    if if_softmax:
        softmax = torch.nn.Softmax(dim=1)
        sus_output = softmax(torch.from_numpy(np.array(sus_output))).numpy()
        poison_output = softmax(torch.from_numpy(np.array(poison_output))).numpy()

    sus_diff = sus_output - poison_output

    print('Load clf')
    # load meta-classifier (clf)
    clf = mlp.MLP5(len(sus_diff[0]), 2)
    if device == 'cpu':
        clf.load_state_dict(torch.load(args.clf_dir, map_location=torch.device('cpu')))
    else:
        clf.load_state_dict(torch.load(args.clf_dir))
    clf = clf.to(device)
    print(clf)
    clf.eval()

    # get probability from clf
    print('get probability from clf')

    if if_load_norm or if_load_scale:
        load_path_list = args.clf_dir.split('/')
        load_path_list = load_path_list[:-1]
        load_path = os.path.join(*load_path_list)


    if if_norm:
        if if_load_norm:
            normalizer_path = os.path.join(load_path, 'normalizer.pkl')
            with open(normalizer_path, 'rb') as f:
                normalizer = pickle.load(f)
        else:
            normalizer = Normalizer(norm='l2')
            sus_diff = normalizer.transform(sus_diff)

    if if_scale:
        if if_load_scale:
            scaler_path = os.path.join(load_path, 'scaler.pkl')
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            sus_diff = scaler.transform(sus_diff)
        else:
            scaler = RobustScaler()
            sus_diff = scaler.fit_transform(sus_diff)

    sus_diff = get_prob_pair(clf, sus_diff, device)

    softmax = torch.nn.Softmax(dim=1)
    softmax_sus_diff = softmax(torch.from_numpy(np.array(sus_diff)))

    # (benign, 0) and (vict, 1)
    sus_diff_stolen = softmax_sus_diff[:, 1]
    sus_diff_benign = softmax_sus_diff[:, 0]


    seed = 100
    m = args.sample_num

    print('sus_diff_stolen', sus_diff_stolen)
    print('sus_diff_benign', sus_diff_benign)


    p_list, mu_list = mult_test(sus_diff_stolen.numpy(), sus_diff_benign.numpy(), seed=seed, m=m, mult_num=40, alternative=args.alternative)
    print('result:  p-val: {} mu: {}'.format(hmean(p_list), np.mean(mu_list)))

    print('Time cost: {} sec'.format(round(time.time() - start_time, 2)))

