import os
import argparse
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from utils import anom_utils


parser = argparse.ArgumentParser(description='Present OOD Detection metrics for Energy-score')
parser.add_argument('--name', '-n', required = True, type=str,
                    help='name of experiment')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset e.g. pascal')
parser.add_argument('--test_epochs', "-e", default = "100", type=str,
                    help='# epoch to test performance')
parser.add_argument('--hist', default = False, type=bool,
                    help='if need to plot histogram')
args = parser.parse_args()

def main():

    if args.in_dataset == "CIFAR-10" or args.in_dataset == "CIFAR-100":
        out_datasets = ['places365','LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'SVHN']
    fprs = dict()
    for test_epoch in args.test_epochs.split():
        all_results_ntom = []
        save_dir =  f"./energy_results/{args.in_dataset}/{args.name}" 
        with open(os.path.join(save_dir, f'energy_score_at_epoch_{test_epoch}.npy'), 'rb') as f:
            id_sum_energy = np.load(f)
        all_results = defaultdict(int)
        for out_dataset in out_datasets:
            with open(os.path.join(save_dir, f'energy_score_{out_dataset}_at_epoch_{test_epoch}.npy'), 'rb') as f:
                ood_sum_energy = np.load(f)
            auroc, aupr, fpr = anom_utils.get_and_print_results(-1 * id_sum_energy, -1 * ood_sum_energy, f"{out_dataset}", f" Energy Sum at epoch {test_epoch}")
            results = cal_metric(known =  -1 * id_sum_energy, novel = -1* ood_sum_energy, method = "energy sum")
            all_results_ntom.append(results)
            all_results["AUROC"] += auroc
            all_results["AUPR"] += aupr
            all_results["FPR95"] += fpr
            if args.hist:
                fig, (ax1) = plt.subplots(1, 1, figsize=(12,12))
                ax1.hist(-1 * id_sum_energy, 20, density = True, alpha=0.5, label='id')
                ax1.hist(-1 * ood_sum_energy, 20, density = True, alpha=0.5, label='ood')
                ax1.set_ylim(0, 1)
                ax1.legend(loc='upper right')
                ax1.set_title("Energy Sum")

                plt.savefig(f"energy_sum_{out_dataset}.png")
        print("Avg FPR95: ", round(100 * all_results["FPR95"]/len(out_datasets),2))
        print("Avg AUROC: ", round(all_results["AUROC"]/len(out_datasets),4))
        print("Avg AUPR: ", round(all_results["AUPR"]/len(out_datasets),4))
        fprs[test_epoch] = 100 * all_results["FPR95"]/len(out_datasets)

def print_results(results, in_dataset, out_dataset, name, method):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    print('in_distribution: ' + in_dataset)
    print('out_distribution: '+ out_dataset)
    print('Model Name: ' + name)
    print('')

    print(' OOD detection method: ' + method)
    for mtype in mtypes:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100.*results['FPR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['DTERR']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*results['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100.*results['AUOUT']), end='')
    print('')

def cal_metric(known, novel, method):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = dict()

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1.-fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp+fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0]-tp+fp[0]-fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0]-fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])

    return results

def get_curve(known, novel, method):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]

    j = num_k+num_n-1
    for l in range(num_k+num_n-1):
        if all[j] == all[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95

def compute_average_results(all_results):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    avg_results = dict()

    for mtype in mtypes:
        avg_results[mtype] = 0.0

    for results in all_results:
        for mtype in mtypes:
            avg_results[mtype] += results[mtype]

    print("len of all results", float(len(all_results)))
    for mtype in mtypes:
        avg_results[mtype] /= float(len(all_results))

    return avg_results

if __name__ == '__main__':
    main()