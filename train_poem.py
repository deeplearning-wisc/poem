import argparse
import os
import time
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import utils.svhn_loader as svhn

import models.densenet as dn
import models.wideresnet as wn
from neural_linear_opt import NeuralLinear, SimpleDataset
from utils import TinyImages, ImageNet
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='Posterior sampling-based outlier mining with enregy-regularized training')

parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset e.g. CIFAR-10')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture e.g. simplenet densenet')

parser.add_argument('--save-epoch', default= 10, type=int,
                    help='save the model every save_epoch') # freq; save model state_dict()
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)') # print every print-freq batches during training
# ID train & val batch size and OOD train batch size 
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64) used for training id and ood')
# training schedule
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default= 100, type=int,
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                    help='weight decay (default: 0.0001)')
# densenet
parser.add_argument('--layers', default= 100, type=int,
                    help='total number of layers (default: 100) for DenseNet')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
# wideresnet
parser.add_argument('--depth', default=40, type=int,
                    help='depth of wide resnet')
parser.add_argument('--width', default=4, type=int,
                    help='width of resnet')
## network spec
parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--beta', default=1.0, type=float, help='beta for out_loss')
# ood sampling and mining
parser.add_argument('--ood-batch-size', default= 2000, type=int,
                    help='mini-batch size (default: 400) used for ood mining')
parser.add_argument('--pool-size', default= 200, type=int,
                    help='pool size')
#posterior sampling
parser.add_argument('--a0', type=float, default=6.0, help='a0')
parser.add_argument('--b0', type=float, default=6.0, help='b0')
parser.add_argument('--lambda_prior', type=float, default=0.25, help='lambda_prior')
parser.add_argument('--sigma', type=float, default=20, help='control var for weights')
parser.add_argument('--sigma_n', type=float, default=0.5, help='control var for noise')
parser.add_argument('--conf', type=float, default=3.0, help='control ground truth for bayesian linear regression. 2.95--0.05; 3.9--0.98; 4.6 --0.99; 6.9--0.999')
# saving, naming and logging
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--auxiliary-dataset', default='imagenet', 
                    choices=['imagenet','80m_tiny_images', 'partial_imagenet'], type=str, help='which auxiliary dataset to use')
parser.add_argument('--name', required = True, type=str,
                    help='name of experiment')
parser.add_argument('--log_name',
                    help='Name of the Log File', type = str, default = "info.log")
parser.add_argument('--ood_factor', type=float, default= 1,
                 help='ood_dataset_size = len(train_loader.dataset) * ood_factor default = 2.0')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
#Device options
parser.add_argument('--gpu-ids', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--energy_model', default='True', type=bool,
                    help='if use energy model')
parser.add_argument('--debug', default='True', type=bool,
                    help='if in debug mode')
parser.add_argument('--m_in', type=float, default=-25., help='default: -25. margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7., help='default: -7. margin for out-distribution; below this value will be penalized')
parser.add_argument('--energy_beta', default=0.1, type=float, help='beta for energy fine tuning loss')
parser.add_argument('--BUF_SIZE', type= int, default=4, help='# of data points (measured w.r.t. # of epochs) used for posterior update')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

args = parser.parse_args()

directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
save_state_file = os.path.join(directory, 'train_args.txt')
fw = open(save_state_file, 'w')
state = {k: v for k, v in args._get_kwargs()}
print(state, file=fw)
fw.close()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
torch.manual_seed(1)
np.random.seed(1)

def main():
    if args.tensorboard: configure("runs/%s"%(args.name))
    
    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(directory, args.log_name), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler) 

    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])
    state = {k: v for k, v in args._get_kwargs()}
    log.debug(state)

    kwargs = {'num_workers': 4, 'pin_memory': True}

    if args.in_dataset == "CIFAR-10":
        # Data loading code
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        lr_schedule=[50, 75, 90]

        num_classes = 10
        pool_size = args.pool_size

    elif args.in_dataset == "CIFAR-100":
        # Data loading code
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        lr_schedule=[50, 75, 90]

        num_classes = 100
        pool_size = args.pool_size

    elif args.in_dataset == "SVHN":
        # Data loading code
        normalizer = None
        train_loader = torch.utils.data.DataLoader(
            svhn.SVHN('datasets/svhn/', split='train',
                                      transform=transforms.ToTensor(), download=False),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            svhn.SVHN('datasets/svhn/', split='test',
                                  transform=transforms.ToTensor(), download=False),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        args.save_epoch = 5
        lr_schedule=[10, 15, 18]
        pool_size = args.pool_size
        num_classes = 10

    ood_dataset_size = int(len(train_loader.dataset) * args.ood_factor)
    print('OOD Dataset Size: ', ood_dataset_size)

    if args.auxiliary_dataset == '80m_tiny_images':
        ood_loader = torch.utils.data.DataLoader(
            TinyImages(transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(), transforms.ToTensor()])),
                batch_size=args.ood_batch_size, shuffle=False, **kwargs)
    elif args.auxiliary_dataset == 'imagenet':
        ood_loader = torch.utils.data.DataLoader(
            ImageNet(transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(), transforms.ToTensor()])),
                batch_size=args.ood_batch_size, shuffle=False, **kwargs)

    # create model
    if args.model_arch == 'densenet':
        model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer)
    elif args.model_arch == 'wideresnet':
        model = wn.WideResNet(args.depth, num_classes + 1, widen_factor=args.width, dropRate=args.droprate, normalizer=normalizer)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    repr_dim = model.repr_dim
    model = model.cuda()
    bayes_nn = NeuralLinear(args, model, repr_dim, output_dim = 1)

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #Start Training 
    bayes_nn.sample_BDQN()
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, lr_schedule) 
        selected_ood_loader = select_ood_opt(ood_loader, model, bayes_nn, args.batch_size * args.ood_factor, 
                                            num_classes, pool_size, ood_dataset_size)
        bayes_nn.train_blr(train_loader, selected_ood_loader, criterion, optimizer, epoch, directory, log, args.energy_model)
        bayes_nn.update_representation()
        bayes_nn.update_bays_reg_BDQN(log)
        bayes_nn.sample_BDQN()
        # evaluate on validation set
        prec1 =  bayes_nn.validate(val_loader, model, criterion, epoch, log, args.energy_model)
        # remember best prec@1 and save checkpoint
        if  (epoch + 1) % args.save_epoch == 0 and (epoch + 1) >= 80:
            # data parallel save
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, epoch + 1)


def select_ood_opt(ood_loader, model, ood_branch, batch_size, num_classes, pool_size, ood_dataset_size):

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    offset = np.random.randint(len(ood_loader.dataset))
    while offset>=0 and offset<10000:
        offset = np.random.randint(len(ood_loader.dataset))
    ood_loader.dataset.offset = offset
    out_iter = iter(ood_loader)
    print('Start selecting OOD samples...')
    # select ood samples
    model.eval()
    with torch.no_grad():
        all_ood_input = torch.empty(0,3,32,32)
        all_abs_val = torch.empty(0)
        duration = 0
        init_start = time.time()
        for k in range(pool_size): 
            start = time.time()
            try:
                out_set = next(out_iter)
            except StopIteration:
                offset = np.random.randint(len(ood_loader.dataset))
                while offset>=0 and offset<10000:
                    offset = np.random.randint(len(ood_loader.dataset))
                ood_loader.dataset.offset = offset
                out_iter = iter(ood_loader)
                out_set = next(out_iter)

            input = out_set[0] 
            output = ood_branch.predict(input.cuda())
            abs_val = torch.abs(output).squeeze() 
            duration += time.time() - start
            all_ood_input = torch.cat((all_ood_input, input), dim = 0)
            all_abs_val = torch.cat((all_abs_val, abs_val.detach().cpu()), dim = 0)
    print('Scanning Time: ',  duration)
    _, selected_indices = torch.topk(all_abs_val, ood_dataset_size, largest=False)
    print('Total OOD samples: ', len(selected_indices))
    ood_images = all_ood_input[selected_indices]
    ood_labels = (torch.ones(ood_dataset_size) * num_classes).long()

    ood_train_loader = torch.utils.data.DataLoader(
        SimpleDataset(ood_images, ood_labels),
        batch_size=batch_size, shuffle=True, num_workers = 2)

    print('Time: ', time.time()-init_start)
    return ood_train_loader

def adjust_learning_rate(optimizer, epoch, lr_schedule=[50, 75, 90]):
    """Sets the learning rate to the initial LR decayed by 10 after 40 and 80 epochs"""
    lr = args.lr
    if epoch >= lr_schedule[0]:
        lr *= 0.1
    if epoch >= lr_schedule[1]:
        lr *= 0.1
    if epoch >= lr_schedule[2]:
        lr *= 0.1
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, epoch):
    """Saves checkpoint to disk"""
    directory = "checkpoints/{in_dataset}/{name}/".format(in_dataset=args.in_dataset, name=args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'checkpoint_{}.pth.tar'.format(epoch)
    torch.save(state, filename)


if __name__ == '__main__':
    main()
