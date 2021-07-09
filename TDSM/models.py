# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
# utils
import math
import os
import numpy as np
import joblib
#from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window,\
                  camel_to_snake  # , FocalLoss


def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault('device', torch.device('cpu'))
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']

    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = kwargs.setdefault('weights', weights)
    kwargs['weights'] = kwargs['weights'].to(device)

    if name == 'baseline':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        model = Baseline(n_bands, n_classes,
                         kwargs.setdefault('dropout', False))
        lr = kwargs.setdefault('learning_rate', 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        # criterion = FocalLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('batch_size', 100)
        # kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[350], gamma=0.1))

    elif name == 'baselineBN':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        model = BaselineBN(n_bands, n_classes, kwargs.setdefault(
            'dropout', False))
        lr = kwargs.setdefault('learning_rate', 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('batch_size', 100)
        # kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[350], gamma=0.1))

    elif name == 'hamida':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = HamidaEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        # Scheduler for learning rate:
        kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[40, 150], gamma=0.1))

    elif name == 'hamidaREG':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = HamidaREG(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.03)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 16384)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        # Scheduler for learning rate:
        kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[40, 150], gamma=0.1))

    elif name == 'lee':
        kwargs.setdefault('epoch', 200)
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = False
        model = LeeEtAl(n_bands, n_classes)
        lr = kwargs.setdefault('learning_rate', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])

    elif name == 'leeBN':
        kwargs.setdefault('epoch', 200)
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = False
        model = LeeBN(n_bands, n_classes)
        lr = kwargs.setdefault('learning_rate', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])

    elif name == 'chen':
        patch_size = kwargs.setdefault('patch_size', 27)
        center_pixel = True
        model = ChenEtAl(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 0.003)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 400)
        kwargs.setdefault('batch_size', 100)

    elif name == 'li':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = LiEtAl(n_bands, n_classes, n_planes=1024,
                       patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 1.0)
        # SGD optim:
        optimizer = optim.SGD(model.parameters(),
                              lr=lr, momentum=0.9, weight_decay=0.0005)
        # optimizer = optim.Adam(model.parameters(),
        #                        lr=lr, weight_decay=0.0005)
        epoch = kwargs.setdefault('epoch', 200)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1))

    elif name == 'liBN':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = LiBN(n_bands, n_classes, n_planes=1024,
                     patch_size=patch_size)
        lr = kwargs.setdefault('learning_rate', 1.0)
        # SGD optim:
        optimizer = optim.SGD(model.parameters(),
                              lr=lr, momentum=0.9, weight_decay=0.0005)
        # optimizer = optim.Adam(model.parameters(),
        #                        lr=lr, weight_decay=0.0005)
        epoch = kwargs.setdefault('epoch', 200)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1))

    elif name == 'hu':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        model = HuEtAl(n_bands, n_classes)
        # From what I infer from the paper (Eq.7 and Algorithm 1),
        # it is standard SGD with lr = 0.01
        lr = kwargs.setdefault('learning_rate', 0.01)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('epoch', 100)
        kwargs.setdefault('batch_size', 100)

    elif name == 'he':
        # We train our model by AdaGrad [18] algorithm, in which
        # the base learning rate is 0.01. In addition, we set the batch
        # as 40, weight decay as 0.01 for all the layers
        # The input of our network is the HSI 3D patch in the size of 7×7×Band
        kwargs.setdefault('patch_size', 7)
        kwargs.setdefault('batch_size', 40)
        lr = kwargs.setdefault('learning_rate', 0.01)
        center_pixel = True
        model = HeEtAl(n_bands, n_classes, patch_size=kwargs['patch_size'])
        # For Adagrad, we need to load the model on GPU before creating
        # the optimizer
        model = model.to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])

    elif name == 'heBN':
        # We train our model by AdaGrad [18] algorithm, in which
        # the base learning rate is 0.01. In addition, we set the batch
        # as 40, weight decay as 0.01 for all the layers
        # The input of our network is the HSI 3D patch in the size of 7×7×Band
        kwargs.setdefault('patch_size', 7)
        kwargs.setdefault('batch_size', 40)
        lr = kwargs.setdefault('learning_rate', 0.01)
        center_pixel = True
        model = HeBN(n_bands, n_classes, patch_size=kwargs['patch_size'])
        # For Adagrad, we need to load the model on GPU before creating
        # the optimizer
        model = model.to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])

    elif name == 'heMod':
        # We train our model by AdaGrad [18] algorithm, in which
        # the base learning rate is 0.01. In addition, we set the batch
        # as 40, weight decay as 0.01 for all the layers
        # The input of our network is the HSI 3D patch in the size of 7×7×Band
        kwargs.setdefault('patch_size', 7)
        kwargs.setdefault('batch_size', 40)
        lr = kwargs.setdefault('learning_rate', 0.01)
        center_pixel = True
        model = HeMod(n_bands, n_classes, patch_size=kwargs['patch_size'])
        # For Adagrad, we need to load the model on GPU before creating
        # the optimizer
        model = model.to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])

    elif name == 'heMod2':
        # We train our model by AdaGrad [18] algorithm, in which
        # the base learning rate is 0.01. In addition, we set the batch
        # as 40, weight decay as 0.01 for all the layers
        # The input of our network is the HSI 3D patch in the size of 7×7×Band
        kwargs.setdefault('patch_size', 7)
        kwargs.setdefault('batch_size', 40)
        lr = kwargs.setdefault('learning_rate', 0.01)
        center_pixel = True
        model = HeMod2(n_bands, n_classes, patch_size=kwargs['patch_size'])
        # For Adagrad, we need to load the model on GPU before creating
        # the optimizer
        model = model.to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])

    elif name == 'heMod3':
        # We train our model by AdaGrad [18] algorithm, in which
        # the base learning rate is 0.01. In addition, we set the batch
        # as 40, weight decay as 0.01 for all the layers
        # The input of our network is the HSI 3D patch in the size of 7×7×Band
        kwargs.setdefault('patch_size', 7)
        kwargs.setdefault('batch_size', 40)
        lr = kwargs.setdefault('learning_rate', 0.01)
        center_pixel = True
        model = HeMod3(n_bands, n_classes, patch_size=kwargs['patch_size'])
        # For Adagrad, we need to load the model on GPU before creating
        # the optimizer
        model = model.to(device)
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])

    elif name == 'luo':
        # All  the  experiments  are  settled  by  the  learning  rate  of
        #  0.1, the  decay  term  of  0.09  and  batch  size  of  100.
        kwargs.setdefault('patch_size', 3)
        kwargs.setdefault('batch_size', 100)
        lr = kwargs.setdefault('learning_rate', 0.1)
        center_pixel = True
        model = LuoEtAl(n_bands, n_classes, patch_size=kwargs['patch_size'])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.09)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])

    elif name == 'sharma':
        # We train our S-CNN from scratch using stochastic gradient descent
        # with momentum set to 0.9, weight decay of 0.0005,
        # and with a batch size of 60.
        # We initialize an equal learning rate for all trainable layers
        # to 0.05, which is manually decreased by a factor of 10 when the
        # validation error stopped decreasing.
        # Prior to the termination the learning rate was
        # reduced two times at 15th and 25th epoch. [...]
        # We trained the network for 30 epochs
        kwargs.setdefault('batch_size', 60)
        epoch = kwargs.setdefault('epoch', 30)
        lr = kwargs.setdefault('learning_rate', 0.05)
        center_pixel = True
        # We assume patch_size = 64
        kwargs.setdefault('patch_size', 64)
        model = SharmaEtAl(n_bands, n_classes, patch_size=kwargs['patch_size'])
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
        kwargs.setdefault('scheduler', optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[epoch // 2, (5 * epoch) // 6], gamma=0.1))

    elif name == 'liu':
        kwargs['supervision'] = 'semi'
        # "The learning rate is set to 0.001 empirically.
        # The number of epochs is set to be 40."
        kwargs.setdefault('epoch', 40)
        lr = kwargs.setdefault('learning_rate', 0.001)
        center_pixel = True
        patch_size = kwargs.setdefault('patch_size', 9)
        model = LiuEtAl(n_bands, n_classes, patch_size)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        # "The unsupervised cost is the squared error of the difference"
        criterion = (nn.CrossEntropyLoss(
            weight=kwargs['weights']),
            lambda rec, data: F.mse_loss(
                rec, data[:, :, :, patch_size//2, patch_size//2].squeeze()))

    elif name == 'boulch':
        kwargs['supervision'] = 'semi'
        kwargs.setdefault('patch_size', 1)
        kwargs.setdefault('epoch', 100)
        lr = kwargs.setdefault('learning_rate', 0.001)
        center_pixel = True
        model = BoulchEtAl(n_bands, n_classes)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = (nn.CrossEntropyLoss(
            weight=kwargs['weights']),
                     lambda rec, data: F.mse_loss(rec, data.squeeze()))

    elif name == 'mou':
        kwargs.setdefault('patch_size', 1)
        center_pixel = True
        kwargs.setdefault('epoch', 100)
        # "The RNN was trained with the Adadelta algorithm [...]
        # We made use of a fairly  high  learning  rate  of  1.0  instead
        # of  the  relatively  low default of  0.002 to  train the  network"
        lr = kwargs.setdefault('learning_rate', 1.0)
        model = MouEtAl(n_bands, n_classes)
        # For Adadelta, we need to load the model on GPU before creating
        # the optimizer
        model = model.to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])

    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    epoch = kwargs.setdefault('epoch', 100)
    kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau
                      (optimizer, factor=0.1, threshold=0.01,
                       patience=10, verbose=True))
    # kwargs.setdefault('scheduler', None)
    kwargs.setdefault('batch_size', 100)
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('flip_augmentation', False)
    kwargs.setdefault('radiation_augmentation', False)
    kwargs.setdefault('mixture_augmentation', False)
    kwargs['center_pixel'] = center_pixel
    return model, optimizer, criterion, kwargs


class Baseline(nn.Module):
    """
    Baseline network
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        # self.fc1 = nn.Linear(input_channels, 2048)
        # self.fc2 = nn.Linear(2048, 4096)
        # self.fc3 = nn.Linear(4096, 2048)
        # self.fc4 = nn.Linear(2048, n_classes)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048*4)
        self.fc4 = nn.Linear(2048*4, 2048*2)
        self.fc5 = nn.Linear(2048*2, 2048)
        self.fc6 = nn.Linear(2048, n_classes)

        # self.fc1 = nn.Linear(input_channels, 2048)
        # self.fc2 = nn.Linear(2048, 4096)
        # self.fc3 = nn.Linear(4096, 2048*4)
        # self.fc4 = nn.Linear(2048*4, 2048*8)
        # self.fc5 = nn.Linear(2048*8, 2048*4)
        # self.fc6 = nn.Linear(2048*4, 2048*2)
        # self.fc7 = nn.Linear(2048*2, 2048)
        # self.fc8 = nn.Linear(2048, n_classes)

        # channel = 1024
        # self.fc1 = nn.Linear(input_channels, channel)
        # self.fc2 = nn.Linear(channel, channel*2)
        # self.fc3 = nn.Linear(channel*2, channel*4)
        # self.fc4 = nn.Linear(channel*4, channel*8)
        # self.fc5 = nn.Linear(channel*8, channel*16)
        # self.fc6 = nn.Linear(channel*16, channel*8)
        # self.fc7 = nn.Linear(channel*8, channel*4)
        # self.fc8 = nn.Linear(channel*4, channel*2)
        # self.fc9 = nn.Linear(channel*2, channel)
        # self.fc10 = nn.Linear(channel, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc3(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = self.fc4(x)

        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc4(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc5(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc6(x)

        # x = F.relu(self.fc1(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc3(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc4(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc5(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc6(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc7(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = self.fc8(x)

        # x = F.relu(self.fc1(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc3(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc4(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc5(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc6(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc7(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc8(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = F.relu(self.fc9(x))
        # if self.use_dropout:
        #     x = self.dropout(x)
        # x = self.fc10(x)

        return x


class BaselineBN(nn.Module):
    """
    Baseline network
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(BaselineBN, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc1_bn = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc2_bn = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, 2048*4)
        self.fc3_bn = nn.BatchNorm1d(2048*4)
        self.fc4 = nn.Linear(2048*4, 2048*2)
        self.fc4_bn = nn.BatchNorm1d(2048*2)
        self.fc5 = nn.Linear(2048*2, 2048)
        self.fc5_bn = nn.BatchNorm1d(2048)
        self.fc6 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc1_bn(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_bn(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc3_bn(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.fc4_bn(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.fc5_bn(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc6(x)

        return x


class Baseline_2(nn.Module):
    """
    Baseline network
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, dropout=False):
        super(Baseline, self).__init__()
        self.use_dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_channels, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048*4)
        self.fc4 = nn.Linear(2048*4, 2048*2)
        self.fc5 = nn.Linear(2048*2, 2048)
        self.fc6 = nn.Linear(2048, n_classes)

        self.apply(self.weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc4(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc5(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc6(x)
        return x


class HuEtAl(nn.Module):
    """
    Deep Convolutional Neural Networks for Hyperspectral Image Classification
    Wei Hu, Yangyu Huang, Li Wei, Fan Zhang and Hengchao Li
    Journal of Sensors, Volume 2015 (2015)
    https://www.hindawi.com/journals/js/2015/258619/
    """
    @staticmethod
    def weight_init(m):
        # [All the trainable parameters in our CNN should be initialized to
        # be a random value between −0.05 and 0.05.]
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            init.uniform_(m.weight, -0.05, 0.05)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def __init__(self, input_channels, n_classes, kernel_size=None,
                 pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
            # [In our experiments, k1 is better to be [ceil](n1/9)]
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            # The authors recommand that k2's value is chosen so that the
            # pooled features have 30~40 values ceil(kernel_size/5) gives
            # the same values as in the paper so let's assume it's okay
            pool_size = math.ceil(kernel_size / 5)
        self.input_channels = input_channels

        # [The first hidden convolution layer C1 filters the n1 x 1 input
        # data with 20 kernels of size k1 x 1]
        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        # [n4 is set to be 100]
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)
        self.apply(self.weight_init)

    def forward(self, x):
        # [In our design architecture, we choose the hyperbolic tangent
        # function tanh(u)]
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = x.view(-1, self.features_size)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3),
                stride=(1, 1, 1),
                dilation=dilation, padding=1)
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3),
                stride=(1, 1, 1),
                dilation=dilation,
                padding=0)
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1),
            dilation=dilation,
            stride=(2, 1, 1),
            padding=(1, 0, 0))
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3),
            dilation=dilation,
            stride=(1, 1, 1),
            padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1),
            dilation=dilation,
            stride=(2, 1, 1),
            padding=(1, 0, 0))
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1),
            dilation=dilation,
            stride=(1, 1, 1),
            padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1),
            dilation=dilation,
            stride=(2, 1, 1),
            padding=(1, 0, 0))

        # self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class HamidaREG(nn.Module):
    """
    Modified architecture with regularization layers of the model from:
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaREG, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3),
                stride=(1, 1, 1),
                dilation=dilation, padding=1)
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3),
                stride=(1, 1, 1),
                dilation=dilation,
                padding=0)
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1),
            dilation=dilation,
            stride=(2, 1, 1),
            padding=(1, 0, 0))
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3),
            dilation=dilation,
            stride=(1, 1, 1),
            padding=(1, 0, 0))
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1),
            dilation=dilation,
            stride=(2, 1, 1),
            padding=(1, 0, 0))
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1),
            dilation=dilation,
            stride=(1, 1, 1),
            padding=(1, 0, 0))
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1),
            dilation=dilation,
            stride=(2, 1, 1),
            padding=(1, 0, 0))

        self.dropout = nn.Dropout(p=0.50)

        self.features_size = self._get_final_flattened_size()
        self.bn = nn.BatchNorm1d(self.features_size)
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.fc(x)
        return x


class LeeEtAl(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes):
        super(LeeEtAl, self).__init__()
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(
            1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv_1x1 = nn.Conv3d(
            1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=0)

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(256, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
        x_3x3 = self.conv_3x3(x)
        x_1x1 = self.conv_1x1(x)
        x = torch.cat([x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.conv1(x)

        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        x_res = F.relu(self.conv2(x))
        x_res = self.conv3(x_res)
        x = F.relu(x + x_res)

        # Second residual block
        x_res = F.relu(self.conv4(x))
        x_res = self.conv5(x_res)
        x = F.relu(x + x_res)

        x = F.relu(self.conv6(x))
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = self.dropout(x)
        x = self.conv8(x)
        return x


class LeeBN(nn.Module):
    """
    CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION
    Hyungtae Lee and Heesung Kwon
    IGARSS 2016
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, in_channels, n_classes):
        super(LeeBN, self).__init__()
        self.bn3 = nn.BatchNorm3d(128)
        self.bn2 = nn.BatchNorm2d(128)
        # The first convolutional layer applied to the input hyperspectral
        # image uses an inception module that locally convolves the input
        # image with two convolutional filters with different sizes
        # (1x1xB and 3x3xB where B is the number of spectral bands)
        self.conv_3x3 = nn.Conv3d(
            1, 128, (in_channels, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.conv_1x1 = nn.Conv3d(
            1, 128, (in_channels, 1, 1), stride=(1, 1, 1), padding=0)

        # We use two modules from the residual learning approach
        # Residual block 1
        self.conv1 = nn.Conv2d(256, 128, (1, 1))
        self.conv2 = nn.Conv2d(128, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, (1, 1))

        # Residual block 2
        self.conv4 = nn.Conv2d(128, 128, (1, 1))
        self.conv5 = nn.Conv2d(128, 128, (1, 1))

        # The layer combination in the last three convolutional layers
        # is the same as the fully connected layers of Alexnet
        self.conv6 = nn.Conv2d(128, 128, (1, 1))
        self.conv7 = nn.Conv2d(128, 128, (1, 1))
        self.conv8 = nn.Conv2d(128, n_classes, (1, 1))

        self.lrn1 = nn.LocalResponseNorm(256)
        self.lrn2 = nn.LocalResponseNorm(128)

        # The 7 th and 8 th convolutional layers have dropout in training
        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def forward(self, x):
        # Inception module
        x_3x3 = self.bn3(self.conv_3x3(x))
        x_1x1 = self.bn3(self.conv_1x1(x))
        x = torch.cat([x_3x3, x_1x1], dim=1)
        # Remove the third dimension of the tensor
        x = torch.squeeze(x)

        # Local Response Normalization
        x = F.relu(self.lrn1(x))

        # First convolution
        x = self.bn2(self.conv1(x))

        # Local Response Normalization
        x = F.relu(self.lrn2(x))

        # First residual block
        x_res = self.bn2(F.relu(self.conv2(x)))
        x_res = self.bn2(self.conv3(x_res))
        x = F.relu(x + x_res)

        # Second residual block
        x_res = self.bn2(F.relu(self.conv4(x)))
        x_res = self.bn2(self.conv5(x_res))
        x = F.relu(x + x_res)

        x = self.bn2(F.relu(self.conv6(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.conv7(x)))
        x = self.dropout(x)
        x = self.conv8(x)
        return x


class ChenEtAl(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON
                        CONVOLUTIONAL NEURAL NETWORKS
    Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia and Pedram Ghamisi
    IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2017
    """
    @staticmethod
    def weight_init(m):
        # In the beginning, the weights are randomly initialized
        # with standard deviation 0.001
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.normal_(m.weight, std=0.001)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=27, n_planes=32):
        super(ChenEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, n_planes, (32, 4, 4))
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(n_planes, n_planes, (32, 4, 4))

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x


class LiEtAl(nn.Module):
    """
    SPECTRAL–SPATIAL CLASSIFICATION OF HYPERSPECTRAL IMAGERY
            WITH 3D CONVOLUTIONAL NEURAL NETWORK
    Ying Li, Haokui Zhang and Qiang Shen
    MDPI Remote Sensing, 2017
    http://www.mdpi.com/2072-4292/9/1/67
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5):
        super(LiEtAl, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        # for the Pavia University and Indian Pines scenes, those in C1 and C2
        # were set to seven and three, respectively
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes,
                               (3, 3, 3), padding=(1, 0, 0))
        # self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class LiBN(nn.Module):
    """
    SPECTRAL–SPATIAL CLASSIFICATION OF HYPERSPECTRAL IMAGERY
            WITH 3D CONVOLUTIONAL NEURAL NETWORK
    Ying Li, Haokui Zhang and Qiang Shen
    MDPI Remote Sensing, 2017
    http://www.mdpi.com/2072-4292/9/1/67
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5):
        super(LiBN, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size
        self.bn1 = nn.BatchNorm3d(n_planes)
        self.bn2 = nn.BatchNorm3d(2 * n_planes)

        # The proposed 3D-CNN model has two 3D convolution layers (C1 and C2)
        # and a fully-connected layer (F1)
        # we fix the spatial size of the 3D convolution kernels to 3 × 3
        # while only slightly varying the spectral depth of the kernels
        # for the Pavia University and Indian Pines scenes, those in C1 and C2
        # were set to seven and three, respectively
        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        # the number of kernels in the second convolution layer is set to be
        # twice as many as that in the first convolution layer
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes,
                               (3, 3, 3), padding=(1, 0, 0))
        # self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.bn1(self.conv1(x))
            x = self.bn2(self.conv2(x))
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = x.view(-1, self.features_size)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


class HeEtAl(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(HeEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3),
                               stride=(3, 1, 1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1),
                                 padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1),
                                 padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1),
                                 padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1),
                                 padding=(5, 0, 0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1),
                                 padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1),
                                 padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1),
                                 padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1),
                                 padding=(5, 0, 0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))
        # the ratio of dropout is 0.6 in our experiments
        self.dropout = nn.Dropout(p=0.6)

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class HeBN(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION + Batch Norm
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(HeBN, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3),
                               stride=(3, 1, 1))
        self.bn = nn.BatchNorm3d(16)
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1),
                                 padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1),
                                 padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1),
                                 padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1),
                                 padding=(5, 0, 0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1),
                                 padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1),
                                 padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1),
                                 padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1),
                                 padding=(5, 0, 0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))
        # the ratio of dropout is 0.6 in our experiments
        self.dropout = nn.Dropout(p=0.6)

        self.features_size = self._get_final_flattened_size()

        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.bn(self.conv1(x))
            x2_1 = self.bn(self.conv2_1(x))
            x2_2 = self.bn(self.conv2_2(x))
            x2_3 = self.bn(self.conv2_3(x))
            x2_4 = self.bn(self.conv2_4(x))
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.bn(self.conv3_1(x))
            x3_2 = self.bn(self.conv3_2(x))
            x3_3 = self.bn(self.conv3_3(x))
            x3_4 = self.bn(self.conv3_4(x))
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.bn(self.conv4(x))
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2_1 = self.bn(self.conv2_1(x))
        x2_2 = self.bn(self.conv2_2(x))
        x2_3 = self.bn(self.conv2_3(x))
        x2_4 = self.bn(self.conv2_4(x))
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = self.bn(F.relu(x))
        x3_1 = self.bn(self.conv3_1(x))
        x3_2 = self.bn(self.conv3_2(x))
        x3_3 = self.bn(self.conv3_3(x))
        x3_4 = self.bn(self.conv3_4(x))
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = self.bn(F.relu(x))
        x = self.bn(F.relu(self.conv4(x)))
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class HeMod(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(HeMod, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3),
                               stride=(3, 1, 1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1),
                                 padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1),
                                 padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1),
                                 padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1),
                                 padding=(5, 0, 0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1),
                                 padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1),
                                 padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1),
                                 padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1),
                                 padding=(5, 0, 0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))

        self.features_size = self._get_final_flattened_size()
        self.bn = nn.BatchNorm1d(self.features_size)
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.bn(x)
        x = self.fc(x)
        return x


class HeMod2(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(HeMod2, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3),
                               stride=(3, 1, 1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1),
                                 padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1),
                                 padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1),
                                 padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1),
                                 padding=(5, 0, 0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1),
                                 padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1),
                                 padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1),
                                 padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1),
                                 padding=(5, 0, 0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))

        self.features_size = self._get_final_flattened_size()
        self.bn1 = nn.BatchNorm1d(self.features_size)
        self.fc1 = nn.Linear(self.features_size, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.fc2(x)
        return x


class HeMod3(nn.Module):
    """
    MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL
    IMAGE CLASSIFICATION
    Mingyi He, Bo Li, Huahui Chen
    IEEE International Conference on Image Processing (ICIP) 2017
    https://ieeexplore.ieee.org/document/8297014/
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_uniform(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=7):
        super(HeMod3, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 16, (11, 3, 3),
                               stride=(3, 1, 1))
        self.conv2_1 = nn.Conv3d(16, 16, (1, 1, 1),
                                 padding=(0, 0, 0))
        self.conv2_2 = nn.Conv3d(16, 16, (3, 1, 1),
                                 padding=(1, 0, 0))
        self.conv2_3 = nn.Conv3d(16, 16, (5, 1, 1),
                                 padding=(2, 0, 0))
        self.conv2_4 = nn.Conv3d(16, 16, (11, 1, 1),
                                 padding=(5, 0, 0))
        self.conv3_1 = nn.Conv3d(16, 16, (1, 1, 1),
                                 padding=(0, 0, 0))
        self.conv3_2 = nn.Conv3d(16, 16, (3, 1, 1),
                                 padding=(1, 0, 0))
        self.conv3_3 = nn.Conv3d(16, 16, (5, 1, 1),
                                 padding=(2, 0, 0))
        self.conv3_4 = nn.Conv3d(16, 16, (11, 1, 1),
                                 padding=(5, 0, 0))
        self.conv4 = nn.Conv3d(16, 16, (3, 2, 2))
        self.pooling = nn.MaxPool2d((3, 2, 2), stride=(3, 2, 2))

        self.features_size = self._get_final_flattened_size()
        self.bn1 = nn.BatchNorm1d(self.features_size)
        self.fc1 = nn.Linear(self.features_size, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            x2_1 = self.conv2_1(x)
            x2_2 = self.conv2_2(x)
            x2_3 = self.conv2_3(x)
            x2_4 = self.conv2_4(x)
            x = x2_1 + x2_2 + x2_3 + x2_4
            x3_1 = self.conv3_1(x)
            x3_2 = self.conv3_2(x)
            x3_3 = self.conv3_3(x)
            x3_4 = self.conv3_4(x)
            x = x3_1 + x3_2 + x3_3 + x3_4
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)
        x = x2_1 + x2_2 + x2_3 + x2_4
        x = F.relu(x)
        x3_1 = self.conv3_1(x)
        x3_2 = self.conv3_2(x)
        x3_3 = self.conv3_3(x)
        x3_4 = self.conv3_4(x)
        x = x3_1 + x3_2 + x3_3 + x3_4
        x = F.relu(x)
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = self.fc3(x)
        return x


class LuoEtAl(nn.Module):
    """
    HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image
    Yanan Luo, Jie Zou, Chengfei Yao, Tao Li, Gang Bai
    International Conference on Pattern Recognition 2018
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=3, n_planes=90):
        super(LuoEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.n_planes = n_planes

        # the 8-neighbor pixels [...] are fed into the Conv1 convolved by
        # n1 kernels and s1 stride. Conv1 results are feature vectors each
        # with height of and the width is 1. After reshape layer, the feature
        # vectors becomes an image-like 2-dimension data.
        # Conv2 has 64 kernels size of 3x3, with stride s2.
        # After that, the 64 results are drawn into a vector as the input of
        # the fully connected layer FC1 which has n4 nodes.
        # In the four datasets, the kernel height nk1 is 24 and stride s1,
        # s2 is 9 and 1
        self.conv1 = nn.Conv3d(1, 90, (24, 3, 3), padding=0, stride=(9, 1, 1))
        self.conv2 = nn.Conv2d(1, 64, (3, 3), stride=(1, 1))

        self.features_size = self._get_final_flattened_size()

        self.fc1 = nn.Linear(self.features_size, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.conv1(x)
            b = x.size(0)
            x = x.view(b, 1, -1, self.n_planes)
            x = self.conv2(x)
            _, c, w, h = x.size()
        return c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        b = x.size(0)
        x = x.view(b, 1, -1, self.n_planes)
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SharmaEtAl(nn.Module):
    """
    HYPERSPECTRAL CNN FOR IMAGE CLASSIFICATION & BAND SELECTION,
    WITH APPLICATION TO FACE RECOGNITION
    Vivek Sharma, Ali Diba, Tinne Tuytelaars, Luc Van Gool
    Technical Report, KU Leuven/ETH Zürich
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=64):
        super(SharmaEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # An input image of size 263x263 pixels is fed to conv1
        # with 96 kernels of size 6x6x96 with a stride of 2 pixels
        self.conv1 = nn.Conv3d(1, 96, (input_channels, 6, 6), stride=(1, 2, 2))
        self.conv1_bn = nn.BatchNorm3d(96)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        #  256 kernels of size 3x3x256 with a stride of 2 pixels
        self.conv2 = nn.Conv3d(1, 256, (96, 3, 3), stride=(1, 2, 2))
        self.conv2_bn = nn.BatchNorm3d(256)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        # 512 kernels of size 3x3x512 with a stride of 1 pixel
        self.conv3 = nn.Conv3d(1, 512, (256, 3, 3), stride=(1, 1, 1))
        # Considering those large kernel values, I assume they actually merge
        # the 3D tensors at each step

        self.features_size = self._get_final_flattened_size()

        # The fc1 has 1024 outputs, where dropout was applied after
        # fc1 with a rate of 0.5
        self.fc1 = nn.Linear(self.features_size, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = F.relu(self.conv1_bn(self.conv1(x)))
            x = self.pool1(x)
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t*c, w, h)
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = self.pool2(x)
            b, t, c, w, h = x.size()
            x = x.view(b, 1, t*c, w, h)
            x = F.relu(self.conv3(x))
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool1(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t*c, w, h)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)
        b, t, c, w, h = x.size()
        x = x.view(b, 1, t*c, w, h)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.features_size)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LiuEtAl(nn.Module):
    """
    A semi-supervised convolutional neural network for hyperspectral image
    classification Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu,
    Zhixiang Xue
    Remote Sensing Letters, 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=9):
        super(LiuEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands
        # for the convolutional "and pooling layer" -> actually 3x3 2D
        # convolutions with B1 outputs
        # "the value of B1 is set to be 80"
        self.conv1 = nn.Conv2d(input_channels, 80, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv1_bn = nn.BatchNorm2d(80)

        self.features_sizes = self._get_sizes()

        self.fc_enc = nn.Linear(self.features_sizes[2], n_classes)

        # Decoder
        self.fc1_dec = nn.Linear(self.features_sizes[2],
                                 self.features_sizes[2])
        self.fc1_dec_bn = nn.BatchNorm1d(self.features_sizes[2])
        self.fc2_dec = nn.Linear(self.features_sizes[2],
                                 self.features_sizes[1])
        self.fc2_dec_bn = nn.BatchNorm1d(self.features_sizes[1])
        self.fc3_dec = nn.Linear(self.features_sizes[1],
                                 self.features_sizes[0])
        self.fc3_dec_bn = nn.BatchNorm1d(self.features_sizes[0])
        self.fc4_dec = nn.Linear(self.features_sizes[0], input_channels)

        self.apply(self.weight_init)

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels,
                         self.patch_size, self.patch_size))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        _, c, w, h = x.size()
        size0 = c * w * h

        x = self.pool1(x)
        _, c, w, h = x.size()
        size1 = c * w * h

        x = self.conv1_bn(x)
        _, c, w, h = x.size()
        size2 = c * w * h

        return size0, size1, size2

    def forward(self, x):
        x = x.squeeze()
        x_conv1 = self.conv1_bn(self.conv1(x))
        x = x_conv1
        x_pool1 = self.pool1(x)
        x = x_pool1
        x_enc = F.relu(x).view(-1, self.features_sizes[2])
        x = x_enc

        x_classif = self.fc_enc(x)

        # x = F.relu(self.fc1_dec_bn(self.fc1_dec(x) + x_enc))
        x = F.relu(self.fc1_dec(x))
        x = F.relu(self.fc2_dec_bn(self.fc2_dec(x) +
                                   x_pool1.view(-1, self.features_sizes[1])))
        x = F.relu(self.fc3_dec_bn(self.fc3_dec(x) +
                                   x_conv1.view(-1, self.features_sizes[0])))
        x = self.fc4_dec(x)
        return x_classif, x


class BoulchEtAl(nn.Module):
    """
    Autoencodeurs pour la visualisation d'images hyperspectrales
    A.Boulch, N. Audebert, D. Dubucq
    GRETSI 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, planes=16):
        super(BoulchEtAl, self).__init__()
        self.input_channels = input_channels
        self.aux_loss_weight = 0.1

        encoder_modules = []
        n = input_channels
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            print(x.size())
            while(n > 1):
                print("---------- {} ---------".format(n))
                if n == input_channels:
                    p1, p2 = 1, 2 * planes
                elif n == input_channels // 2:
                    p1, p2 = 2 * planes, planes
                else:
                    p1, p2 = planes, planes
                encoder_modules.append(nn.Conv1d(p1, p2, 3, padding=1))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.MaxPool1d(2))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.ReLU(inplace=True))
                x = encoder_modules[-1](x)
                print(x.size())
                encoder_modules.append(nn.BatchNorm1d(p2))
                x = encoder_modules[-1](x)
                print(x.size())
                n = n // 2

            encoder_modules.append(nn.Conv1d(planes, 3, 3, padding=1))
        encoder_modules.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_modules)
        self.features_sizes = self._get_sizes()

        self.classifier = nn.Linear(self.features_sizes, n_classes)
        self.regressor = nn.Linear(self.features_sizes, input_channels)
        self.apply(self.weight_init)

    def _get_sizes(self):
        with torch.no_grad():
            x = torch.zeros((10, 1, self.input_channels))
            x = self.encoder(x)
            _, c, w = x.size()
        return c*w

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(-1, self.features_sizes)
        x_classif = self.classifier(x)
        x = self.regressor(x)
        return x_classif, x


class MouEtAl(nn.Module):
    """
    Deep recurrent neural networks for hyperspectral image classification
    Lichao Mou, Pedram Ghamisi, Xiao Xang Zhu
    https://ieeexplore.ieee.org/document/7914752/
    """
    @staticmethod
    def weight_init(m):
        # All weight matrices in our RNN and bias vectors are initialized with
        # a uniform distribution, and the values of these weight matrices and
        # bias vectors are initialized in the range [−0.1,0.1]
        if isinstance(m, (nn.Linear, nn.GRU)):
            init.uniform_(m.weight.data, -0.1, 0.1)
            init.uniform_(m.bias.data, -0.1, 0.1)

    def __init__(self, input_channels, n_classes):
        # The proposed network model uses a single recurrent layer that adopts
        # our modified GRUs of size 64 with sigmoid gate activation and
        # PRetanh activation functions for hidden representations
        super(MouEtAl, self).__init__()
        self.input_channels = input_channels
        # TODO: try to change this?:
        self.gru = nn.GRU(1, 64, 1, bidirectional=False)
        self.gru_bn = nn.BatchNorm1d(64*input_channels)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(64*input_channels, n_classes)

    def forward(self, x):
        x = x.squeeze()
        x = x.unsqueeze(0)
        # x is in 1, N, C but we expect C, N, 1 for GRU layer
        x = x.permute(2, 1, 0)
        x = self.gru(x)[0]
        # x is in C, N, 64, we permute back
        x = x.permute(1, 2, 0).contiguous()
        x = x.view(x.size(0), -1)
        x = self.gru_bn(x)
        x = self.tanh(x)
        x = self.fc(x)
        return x


def train(net, optimizer, criterion, data_loader, hyperparams, exp_name,
          cross_dict=None, monitor_train=None, last_epoch=0,
          display_iter=10, display=None, val_loader=None,
          validation_metric='Accuracy'):
    """
    Training loop to optimize a network for several epochs and a specified loss
    Args:
    - net (nn.Module): PyTorch model
    - optimizer (nn.Optimizer): PyTorch optimizer
    - criterion (nn.Criterion): PyTorch-compatible loss function,
    e.g. nn.CrossEntropyLoss
    - data_loader (data.DataLoader): PyTorch training dataset loader
    - hyperparams (dict): dictionary containing hyperparameters
    - exp_name (str): experiment name
    - cross_dict (dict): dictionary containing cross_validation info
    - monitor_train (dict): dictionary containing training monitoring summary
    - last_epoch (int): last epoch index
    - display_iter (int): number of iterations before refreshing the display
    - display (bool): activate display
    - val_loader (data.DataLoader): Pytorch validation dataset loader
    Kwargs:
    - validation_metric (str): metric used to update scheduler & save best
    model, chose among: {'Accuracy','Balanced Accuracy', 'Loss'}
    Returns:
    - best_path (str): path to the best checkpoint saved
    """
    epoch = hyperparams['epoch']
    scheduler = hyperparams['scheduler']
    device = hyperparams['device']
    supervision = hyperparams['supervision']

    hyperparams.update({'validation_metric': validation_metric})
    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)
    loss_win, val_win = None, [None]*3

    save_epoch = 5 if epoch > 5 else 1

    if monitor_train is None:
        monitor_train = {
            'losses': np.zeros(len(data_loader)*epoch+1),
            'mean_losses': np.zeros(len(data_loader)*epoch+1),
            'val_metrics': {
                'Accuracy': {'list': [], 'best': 0},
                'Balanced Accuracy': {'list': [], 'best': 0},
                'Loss': {'list': [], 'best': 0}}}
    else:
        # Regenerate plots from stopped training
        iter_ = last_epoch*len(data_loader)
        loss_win = display.line(
            X=np.arange(iter_),
            Y=monitor_train['mean_losses'][0:iter_],
            win=loss_win,
            opts={'title': "Training loss",
                  'xlabel': "Iterations",
                  'ylabel': "Loss"})
        for i, (curr_metric, metric_dict) in enumerate(
                    monitor_train['val_metrics'].items()):
            val_win[i] = display.line(
                Y=np.array(metric_dict['list']),
                X=np.arange(len(metric_dict['list'])),
                win=val_win[i],
                opts={
                    'title': "Validation {}".format(curr_metric),
                    'xlabel': "Epochs",
                    'ylabel': curr_metric})

    print('\n')
    for e in tqdm(range(last_epoch+1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        iter_ = (e-1)*len(data_loader) + 1

        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(enumerate(data_loader),
                                              total=len(data_loader)):
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            # Compute loss
            optimizer.zero_grad()
            if supervision == 'full':
                output = net(data)
                loss = criterion(output, target)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
                loss = criterion[0](output, target)+net.aux_loss_weight * \
                    criterion[1](rec, data)
            else:
                raise ValueError(
                    "supervision mode \"{}\" unknown.".format(supervision))

            # Update weights
            loss.backward()
            optimizer.step()

            # Save loss value
            monitor_train['losses'][iter_] = loss.item()
            monitor_train['mean_losses'][iter_] = np.mean(
                monitor_train['losses'][max(1, iter_ - 10):iter_ + 1])

            # Display loss
            if display_iter and iter_ % display_iter == 0:
                string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                string = string.format(
                    e, epoch, batch_idx *
                    len(data), len(data) * len(data_loader),
                    100. * batch_idx / len(data_loader),
                    monitor_train['mean_losses'][iter_])
                update = None if loss_win is None else 'append'
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=monitor_train['mean_losses'][iter_ - display_iter:iter_],
                    win=loss_win,
                    update=update,
                    opts={'title': "Training loss",
                          'xlabel': "Iterations",
                          'ylabel': "Loss"})
                tqdm.write(string)

            iter_ += 1
            del(data, target, loss, output)

        # Run a test on the validation set
        if val_loader is not None:
            val_metrics = val(net, val_loader,
                              criterion=criterion,
                              device=device,
                              supervision=supervision)
            for i, (curr_metric, metric_dict) in enumerate(
                    monitor_train['val_metrics'].items()):
                metric_dict['list'].append(val_metrics[i])
                if validation_metric == curr_metric:
                    metric = metric_dict['list'][-1]
                    is_best = (metric_dict['best'] < metric)
                # Display validation metrics
                val_win[i] = display.line(
                    Y=np.array(metric_dict['list']),
                    X=np.arange(len(metric_dict['list'])),
                    win=val_win[i],
                    opts={
                        'title': "Validation {}".format(curr_metric),
                        'xlabel': "Epochs",
                        'ylabel': curr_metric})

        # Update the scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the model checkpoint with best performances
        if is_best:
            monitor_train['val_metrics']['Accuracy']['best'] = metric
            best_checkpoint = {
                'epoch': e,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'monitor_train': monitor_train}
            best_path = save_model(net, camel_to_snake(
                str(net.__class__.__name__)),
                                   data_loader.dataset.name,
                                   exp_name, cross_dict,
                                   is_best=True,
                                   optimizer_ckpt=True,
                                   checkpoint=best_checkpoint)

        # Save checkpoint periodically
        checkpoint = {
            'epoch': e,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'monitor_train': monitor_train}
        if e % save_epoch == 0:
            _ = save_model(net, camel_to_snake(str(net.__class__.__name__)),
                           data_loader.dataset.name,
                           exp_name, cross_dict,
                           is_best=False,
                           optimizer_ckpt=True,
                           checkpoint=checkpoint)

    # Save final checkpoint
    _ = save_model(net, camel_to_snake(str(net.__class__.__name__)),
                   data_loader.dataset.name,
                   exp_name, cross_dict,
                   is_best=False,
                   optimizer_ckpt=True,
                   checkpoint=checkpoint,
                   finished=True)

    print('Best model obtained at {} epoch (metric: {})'.format(
        best_checkpoint['epoch'], validation_metric))
    return best_path


def save_model(model, model_name, dataset_name, exp_name, cross_dict,
               is_best=False, optimizer_ckpt=True, checkpoint=None,
               finished=False):
    """ Save current model state.
    Args:
    - model (sklearn.SVC or nn.Module): model to be saved
    - mode_name (str): model name
    - dataset_name (str): dataset name
    - exp_name (str): experiment name
    - cross_dict (dict): ditionary containing cross validation infos
    - is_best (bool): specify if this is the current best model
    - optimizer_ckpt(bool): specify if we save the optimizer params
    - checkpoint(dict): dictionary containing the model and optimizer
                        checkpoints
    - finished(bool): overwrite last checkpoint saved with the final one
    Returns:
    - best_path (str): path to the checkpoint to be used for testing
    """
    # Create checkpoint directory
    model_dir = os.path.join(exp_name, 'checkpoint')
    if cross_dict:
        cv_step = cross_dict['cv_step']
        model_dir = os.path.join(model_dir, "k" + str(cv_step+1))
    os.makedirs(model_dir, exist_ok=True)

    # Generate Chkpt name:
    if is_best:
        filename = 'BEST_' + model_name
    else:
        filename = 'LAST_' + model_name

    # Save chkpt:
    if isinstance(model, torch.nn.Module):
        model_path = os.path.join(model_dir, filename+'.pth')
        if finished:
            tqdm.write("Remove backed-up checkpoint and save final weights.")
            os.replace(model_path, model_path.replace('LAST', 'FINAL'))
            model_path = model_path.replace('LAST', 'FINAL')

        if optimizer_ckpt:
            torch.save(checkpoint, model_path)
        else:
            torch.save(model.state_dict(), model_path)
    else:
        tqdm.write("Saving model params in {}".format(filename))
        model_path = os.path.join(model_dir, filename+'.pkl')
        joblib.dump(model, model_path)

    return model_path


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    softmax = nn.Softmax(dim=1)

    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    kwargs = {'step': hyperparams['test_stride'],
              'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    print('\n')
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)
            output = softmax(output)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs


def val(net, data_loader, criterion, device='cpu', supervision='full',
        metric='accuracy', ):
    """ Compute different metrcis on validation set.
    Args:
    - net (nn.Module): pytorch model
    - data_loader (data.DataLoader): validation dataset
    - criterion (nn.Criterion): loss criterion
    - device (str): device to use
    - supervision (str): supervision mode
    Returns:
    - accuracy (float): accuracy
    - balanced_accuracy (float): balanced accuracy
    - avg_loss (float): average loss
    """
    prediction, gd_truth, loss = [], [], []
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            if supervision == 'full':
                output = net(data)
                loss.append(criterion(output, target).item())
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
                loss.append(
                    criterion[0](output, target) + net.aux_loss_weight *
                    criterion[1](rec, data).item())

            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    prediction.append(pred.item())
                    gd_truth.append(out.item())

    accuracy = accuracy_score(gd_truth, prediction)
    balanced_acc = balanced_accuracy_score(gd_truth, prediction)
    avg_loss = np.mean(np.array(loss))
    return accuracy, balanced_acc, avg_loss
