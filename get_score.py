import argparse
import os
import logging

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision

import utils.svhn_loader as svhn
import models.densenet as dn
import models.wideresnet as wn

parser = argparse.ArgumentParser(description='OOD Detection Evaluation based on Energy-score')

parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture')
parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)') # print every print-freq batches during training
# ID train & val batch size and OOD train batch size 
parser.add_argument('-b', '--batch-size', default= 64, type=int,
                    help='mini-batch size (default: 64) used for training id and ood')
# densenet
parser.add_argument('--layers', default= 100, type=int,
                    help='total number of layers (default: 100) for DenseNet')
parser.add_argument('--growth', default= 12, type=int,
                    help='number of new channels per layer (default: 12)')
# network spec
parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
# ood sampling and mining
parser.add_argument('--ood-batch-size', default= 400, type=int,
                    help='mini-batch size (default: 400) used for testing')
parser.add_argument('--name', '-n', required=True, type=str,
                    help='name of experiment')
parser.add_argument('--test_epochs', "-e", default = "80 90 100", type=str,
                     help='# epoch to test performance')
parser.add_argument('--log_name',
                    help='Name of the Log File', type = str, default = "info_val.log")
parser.add_argument('--base-dir', default='output/ood_scores', type=str, help='result directory')
#Device options
parser.add_argument('--gpu-ids', default='2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

print(state)
directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
torch.manual_seed(1)
np.random.seed(1)
np.random.seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(1)

# devices = list(range(torch.cuda.device_count()))
# Random seed
# if args.manualSeed is None:
#     args.manualSeed = random.randint(1, 10000)
# torch.manual_seed(args.manualSeed)
# np.random.seed(args.manualSeed)
# if use_cuda:
#     torch.cuda.manual_seed_all(args.manualSeed)


def main():

    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(directory, args.log_name), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler) 

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])

    kwargs = {'num_workers': 4, 'pin_memory': True}
    normalizer = transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0))
    if args.in_dataset == "CIFAR-10":
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=False, download = True, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        num_classes = 10
    elif args.in_dataset == "CIFAR-100":
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=False, download = True, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        num_classes = 100

    # create model
    if args.model_arch == 'densenet':
        model = dn.DenseNet3(args.layers, num_classes, normalizer=normalizer)
    elif args.model_arch == 'wideresnet':
        model = wn.WideResNet(args.depth, num_classes, widen_factor=args.width, dropRate=args.droprate, normalizer=normalizer)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    test_epochs = args.test_epochs.split()
    if args.in_dataset == "CIFAR-10" or args.in_dataset == "CIFAR-100":
        out_datasets = ['LSUN', 'places365', 'LSUN_resize', 'iSUN', 'dtd', 'SVHN']

    # load model and store test results
    for test_epoch in test_epochs:
        checkpoint = torch.load("./checkpoints/{in_dataset}/{name}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, name=args.name, epochs= test_epoch))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model.cuda()
        save_dir =  f"./energy_results/{args.in_dataset}/{args.name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print("processing ID")
        id_sum_energy = get_energy(args, model, val_loader, test_epoch, log)
        with open(os.path.join(save_dir, f'energy_score_at_epoch_{test_epoch}.npy'), 'wb') as f:
            np.save(f, id_sum_energy)
        for out_dataset in out_datasets:
            print("processing OOD dataset ", out_dataset)
            testloaderOut = get_ood_loader(out_dataset)
            ood_sum_energy = get_energy(args, model, testloaderOut, test_epoch, log)
            with open(os.path.join(save_dir, f'energy_score_{out_dataset}_at_epoch_{test_epoch}.npy'), 'wb') as f:
                np.save(f, ood_sum_energy)

def get_energy(args, model, val_loader, epoch, log):
    in_energy = AverageMeter()
    model.eval()
    init = True
    log.debug("######## Start collecting energy score ########")
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            # labels = labels.cuda().float()
            outputs = model(images)
            e_s = -torch.logsumexp(outputs, dim=1)
            e_s = e_s.data.cpu().numpy() 
            in_energy.update(e_s.mean(), len(labels))  #DEBUG
            if init:
                sum_energy = e_s
                init = False
            else:
                sum_energy = np.concatenate((sum_energy, e_s))

            if i % args.print_freq == 0:
                log.debug('Epoch: [{0}] Batch#[{1}/{2}]\t'
                    'Energy Sum {in_energy.val:.4f} ({in_energy.avg:.4f})'.format(
                        epoch, i, len(val_loader), in_energy=in_energy))

    return sum_energy

def get_ood_loader(out_dataset):
        if out_dataset == 'SVHN':
            testsetout = svhn.SVHN('datasets/ood_datasets/svhn/', split='test',
                                  transform=transforms.ToTensor(), download=False)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)
        elif out_dataset == "CIFAR-100":
            testloaderOut = torch.utils.data.DataLoader(
                datasets.CIFAR100('./datasets/cifar100', train=False, download=False,
                                transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])
                                ),
            batch_size=args.ood_batch_size, shuffle=True, num_workers=2)
        elif out_dataset == 'dtd':
            testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/dtd/images",
                                        transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size, shuffle=True,
                                                     num_workers=2)
        elif out_dataset == 'places365':
            # root = '/nobackup/dataset_myf/places_subset'
            testsetout = torchvision.datasets.ImageFolder(root="datasets/ood_datasets/places365",
                transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
            subset = torch.utils.data.Subset(testsetout, np.random.choice(len(testsetout), 10000, replace=False))
            testloaderOut = torch.utils.data.DataLoader(subset, batch_size=args.ood_batch_size,
                                                     num_workers=2, shuffle=True)
        else:
            print("Not specified")
            testsetout = torchvision.datasets.ImageFolder("datasets/ood_datasets/{}".format(out_dataset),
                                        transform=transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()]))
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=args.ood_batch_size,
                                             shuffle=True, num_workers=2)
        return testloaderOut

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
