import os
import abc
import math
import json
import torch
import typing
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import pickle
import models.initializers as initializers
import models.bn_initializers as bn_initializers

class Model(abc.ABC, torch.nn.Module):
    """The base class used by all models in this codebase."""

    @staticmethod
    @abc.abstractmethod
    def is_valid_model_name(model_name: str) -> bool:
        """Is the model name string a valid name for models in this class?"""
        pass

    @staticmethod
    @abc.abstractmethod
    def get_model_from_name(
        model_name: str,
        outputs: int,
        initializer: typing.Callable[[torch.nn.Module], None]
    ) -> 'Model':
        """Returns an instance of this class as described by the model_name string."""
        pass

    @property
    def prunable_layer_names(self) -> typing.List[str]:
        """A list of the names of Tensors of this model that are valid for pruning.

        By default, only the weights of convolutional and linear layers are prunable.
        """
        return [name + '.weight' for name, module in self.named_modules() if
                isinstance(module, torch.nn.modules.conv.Conv2d) or
                isinstance(module, torch.nn.modules.linear.Linear)]

    @property
    @abc.abstractmethod
    def output_layer_names(self) -> typing.List[str]:
        """A list of the names of the Tensors of the output layer of this model."""
        pass

    @property
    @abc.abstractmethod
    def loss_criterion(self) -> torch.nn.Module:
        """The loss criterion to use for this model."""
        pass


class Mask(dict):
    def __init__(self, other_dict=None):
        super(Mask, self).__init__()
        if other_dict is not None:
            for k, v in other_dict.items(): self[k] = v

    def __setitem__(self, key, value):
        if not isinstance(key, str) or len(key) == 0:
            raise ValueError('Invalid tensor name: {}'.format(key))
        if isinstance(value, np.ndarray):
            value = torch.as_tensor(value)
        if not isinstance(value, torch.Tensor):
            raise ValueError('value for key {} must be torch Tensor or numpy ndarray.'.format(key))
        if ((value != 0) & (value != 1)).any(): raise ValueError('All entries must be 0 or 1.')

        super(Mask, self).__setitem__(key, value)

    @staticmethod
    def ones_like(model: Model) -> 'Mask':
        mask = Mask()
        for name in model.prunable_layer_names:
            mask[name] = torch.ones(list(model.state_dict()[name].shape))
        return mask

    @staticmethod
    def load(output_location):
        return Mask(torch.load(output_location))

    @staticmethod
    def exists(output_location):
        return True

    def numpy(self):
        return {k: v.cpu().numpy() for k, v in self.items()}

    @property
    def sparsity(self):
        """Return the percent of weights that have been pruned as a decimal."""

        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in self.values()]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in self.values()]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity


class DenseNet(Model):
    class BasicBlock(nn.Module):
        def __init__(self, in_planes, out_planes, dropRate=0.0):
            super(DenseNet.BasicBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                                    padding=1, bias=False)
            self.droprate = dropRate
        def forward(self, x):
            out = self.conv1(self.relu(self.bn1(x)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            return torch.cat([x, out], 1)

    class BottleneckBlock(nn.Module):
        def __init__(self, in_planes, out_planes, dropRate=0.0):
            super(DenseNet.BottleneckBlock, self).__init__()
            inter_planes = out_planes * 4
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, 
                                    padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(inter_planes)
            self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                                    padding=1, bias=False)
            self.droprate = dropRate
        def forward(self, x):
            out = self.conv1(self.relu(self.bn1(x)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
            out = self.conv2(self.relu(self.bn2(out)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
            return torch.cat([x, out], 1)

    class TransitionBlock(nn.Module):
        def __init__(self, in_planes, out_planes, dropRate=0.0):
            super(DenseNet.TransitionBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                                    padding=0, bias=False)
            self.droprate = dropRate
        def forward(self, x):
            out = self.conv1(self.relu(self.bn1(x)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
            return F.avg_pool2d(out, 2)

    class DenseBlock(nn.Module):
        def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
            super(DenseNet.DenseBlock, self).__init__()
            self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
        def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
            layers = []
            for i in range(nb_layers):
                layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
            return nn.Sequential(*layers)
        def forward(self, x):
            return self.layer(x)

    def __init__(self, initializer, depth, num_classes, growth_rate=12, reduction=0.5, 
                    bottleneck=True, dropRate=0.0, normalizer = None, out_classes = 100):
        super(DenseNet, self).__init__()

        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = int(n/2)
            block = DenseNet.BottleneckBlock
        else:
            block = DenseNet.BasicBlock
        
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        # 1st block
        self.block1 = DenseNet.DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = DenseNet.TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        
        # 2nd block
        self.block2 = DenseNet.DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = DenseNet.TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        
        # 3rd block
        self.block3 = DenseNet.DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.criterion = nn.CrossEntropyLoss()

        self.in_planes = in_planes
        self.normalizer = normalizer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        if self.normalizer is not None:
            x = x.clone()
            x[:,0,:,:] = (x[:,0,:,:] - self.normalizer.mean[0]) / self.normalizer.std[0]
            x[:,1,:,:] = (x[:,1,:,:] - self.normalizer.mean[1]) / self.normalizer.std[1]
            x[:,2,:,:] = (x[:,2,:,:] - self.normalizer.mean[2]) / self.normalizer.std[2]

        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        out = self.fc(out)

        return out

    # function to extact the multiple features
    def feature_list(self, x):
        if self.normalizer is not None:
            x = x.clone()
            x[:,0,:,:] = (x[:,0,:,:] - self.normalizer.mean[0]) / self.normalizer.std[0]
            x[:,1,:,:] = (x[:,1,:,:] - self.normalizer.mean[1]) / self.normalizer.std[1]
            x[:,2,:,:] = (x[:,2,:,:] - self.normalizer.mean[2]) / self.normalizer.std[2]

        out_list = []
        out = self.conv1(x)
        out_list.append(out)
        out = self.trans1(self.block1(out))
        out_list.append(out)
        out = self.trans2(self.block2(out))
        out_list.append(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out_list.append(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)

        return self.fc(out), out_list

    def intermediate_forward(self, x, layer_index):
        if self.normalizer is not None:
            x = x.clone()
            x[:,0,:,:] = (x[:,0,:,:] - self.normalizer.mean[0]) / self.normalizer.std[0]
            x[:,1,:,:] = (x[:,1,:,:] - self.normalizer.mean[1]) / self.normalizer.std[1]
            x[:,2,:,:] = (x[:,2,:,:] - self.normalizer.mean[2]) / self.normalizer.std[2]

        out = self.conv1(x)
        if layer_index == 1:
            out = self.trans1(self.block1(out))
        elif layer_index == 2:
            out = self.trans1(self.block1(out))
            out = self.trans2(self.block2(out))
        elif layer_index == 3:
            out = self.trans1(self.block1(out))
            out = self.trans2(self.block2(out))
            out = self.block3(out)
            out = self.relu(self.bn1(out))
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        if self.normalizer is not None:
            x = x.clone()
            x[:,0,:,:] = (x[:,0,:,:] - self.normalizer.mean[0]) / self.normalizer.std[0]
            x[:,1,:,:] = (x[:,1,:,:] - self.normalizer.mean[1]) / self.normalizer.std[1]
            x[:,2,:,:] = (x[:,2,:,:] - self.normalizer.mean[2]) / self.normalizer.std[2]

        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        penultimate = self.relu(self.bn1(out))
        out = F.avg_pool2d(penultimate, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out), penultimate

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return True

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=10):
        num_classes = 10
        depth = 40
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        return DenseNet(initializer=initializer, depth=depth, num_classes=num_classes, normalizer=normalizer)

    @property
    def loss_criterion(self):
        return self.criterion


class PrunedModel(Model):
    @staticmethod
    def to_mask_name(name):
        return 'mask_' + name.replace('.', '___')

    def __init__(self, model: Model, mask: Mask):
        if isinstance(model, PrunedModel): raise ValueError('Cannot nest pruned models.')
        super(PrunedModel, self).__init__()
        self.model = model

        for k in self.model.prunable_layer_names:
            if k not in mask: raise ValueError('Missing mask value {}.'.format(k))
            if not np.array_equal(mask[k].shape, np.array(self.model.state_dict()[k].shape)):
                raise ValueError('Incorrect mask shape {} for tensor {}.'.format(mask[k].shape, k))

        for k in mask:
            if k not in self.model.prunable_layer_names:
                raise ValueError('Key {} found in mask but is not a valid model tensor.'.format(k))

        for k, v in mask.items(): self.register_buffer(PrunedModel.to_mask_name(k), v.float())
        self._apply_mask()

    def _apply_mask(self):
        for name, param in self.model.named_parameters():
            if hasattr(self, PrunedModel.to_mask_name(name)):
                param.data *= getattr(self, PrunedModel.to_mask_name(name))

    def forward(self, x):
        self._apply_mask()
        return self.model.forward(x)

    @property
    def prunable_layer_names(self):
        return self.model.prunable_layer_names

    @property
    def output_layer_names(self):
        return self.model.output_layer_names

    @property
    def loss_criterion(self):
        return self.model.loss_criterion

    def save(self, save_location, save_step):
        self.model.save(save_location, save_step)

    @staticmethod
    def is_valid_model_name(model_name): raise NotImplementedError()

    @staticmethod
    def get_model_from_name(model_name, outputs, initializer): raise NotImplementedError()


def get_model(outputs=None):
    model_name = 'cifar_densenet'
    model_init = 'kaiming_normal'
    batchnorm_init = 'uniform'
    batchnorm_frozen = False
    output_frozen: bool = False
    others_frozen_exceptions = None
    others_frozen = False

    # Select the initializer.
    if hasattr(initializers, model_init):
        initializer = getattr(initializers, model_init)
    else:
        raise ValueError('No initializer: {}'.format(model_init))

    # Select the BatchNorm initializer.
    if hasattr(bn_initializers, batchnorm_init):
        bn_initializer = getattr(bn_initializers, batchnorm_init)
    else:
        raise ValueError('No batchnorm initializer: {}'.format(batchnorm_init))

    # Create the overall initializer function.
    def init_fn(w):
        initializer(w)
        bn_initializer(w)

    # Select the model.
    model = DenseNet.get_model_from_name(model_name, init_fn, outputs)

    if model is None:
        raise ValueError('No such model')

    # Freeze various subsets of the network.
    bn_names = []
    for k, v in model.named_modules():
        if isinstance(v, torch.nn.BatchNorm2d):
            bn_names += [k + '.weight', k + '.bias']

    if others_frozen_exceptions:
        others_exception_names = others_frozen_exceptions.split(',')
        for name in others_exception_names:
            if name not in model.state_dict():
                raise ValueError(f'Invalid name to except: {name}')
    else:
        others_exception_names = []

    for k, v in model.named_parameters():
        if k in bn_names and batchnorm_frozen:
            v.requires_grad = False
        elif k in model.output_layer_names and output_frozen:
            v.requires_grad = False
        elif k not in bn_names and k not in model.output_layer_names and others_frozen:
            if k in others_exception_names: continue
            v.requires_grad = False

    return model




# mask = Mask().load('C:/Users/andre/OneDrive/Desktop/results_densenet/level_0/mask.pth')
# pruned_model = PrunedModel(model, mask)

# -------------------------- Below is where Testing Occurs! -------------------------- #

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)
    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names

