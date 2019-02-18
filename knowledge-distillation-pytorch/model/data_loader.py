"""
   CIFAR-10 data normalization reference:
   https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""

import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from model import cifarsel_dataloader
from torch.utils.data.sampler import SubsetRandomSampler


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def fetch_dataloader(types, params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """

    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    # For CIFAR-10
    # cifar-10 name-class map
    name_class={'airplane':0,'automobile':1,'bird':2,'cat':3,'deer':4,'dog':5,'frog':6,'horse':7,'ship':8,'truck':9}
    name = ['dog','cat','deer','horse']
    name_cifar10_vehicles = ['airplane','automobile','truck']
    name_exp1 = ['dog','cat']
    # for CIFAR100
    insect_name = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'] 
    fruit_name = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']
    #cifar-100 name-class map 
    meta = unpickle('../data/cifar-100-python/meta')
    name_class_cifar100 = {}
    for i,name in enumerate(meta['fine_label_names']):
        name_class_cifar100[name]=i
        
# change the names in the CIFARSel functional call
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    '''
    trainset = cifarsel_dataloader.CIFARSel(root = '../../data',names = name_cifar10_vehicles ,name_class=name_class,train=True,
                             transform = transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.RandomSizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    devset = cifarsel_dataloader.CIFARSel(root = '../../data',names = name_cifar10_vehicles, name_class=name_class,train=False,
                             transform = transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    '''
    
    trainset = cifarsel_dataloader.CIFAR100Sel(root = '../../data', names = insect_name ,name_class=name_class_cifar100,train=True,
                    transform = transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.RandomSizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)

    devset = cifarsel_dataloader.CIFAR100Sel(root = '../../data',names = insect_name, name_class=name_class_cifar100,train=False,
                             transform = transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl


def fetch_subset_dataloader(types, params):
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    """

    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # For CIFAR-10
    # cifar-10 name-class map
    name_class={'airplane':0,'automobile':1,'bird':2,'cat':3,'deer':4,'dog':5,'frog':6,'horse':7,'ship':8,'truck':9}
    name = ['dog','cat','deer','horse']
    name_cifar10_vehicles = ['airplane','automobile','truck']
    name_exp1 = ['dog','cat']
    # for CIFAR100
    insect_name = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'] 
    fruit_name = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']
    #cifar-100 name-class map 
    meta = unpickle('../data/cifar-100-python/meta')
    name_class_cifar100 = {}
    for i,name in enumerate(meta['fine_label_names']):
        name_class_cifar100[name]=i
        
# change the names in the CIFARSel functional call
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    '''
    trainset = cifarsel_dataloader.CIFARSel(root = '../../data',names = name_cifar10_vehicles ,name_class=name_class,train=True,
                             transform = transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.RandomSizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    devset = cifarsel_dataloader.CIFARSel(root = '../../data',names = name_cifar10_vehicles, name_class=name_class,train=False,
                             transform = transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    '''
    
    trainset = cifarsel_dataloader.CIFAR100Sel(root = '../../data', names = insect_name ,name_class=name_class_cifar100,train=True,
                    transform = transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.RandomSizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)

    devset = cifarsel_dataloader.CIFAR100Sel(root = '../../data',names = insect_name, name_class=name_class_cifar100,train=False,
                             transform = transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
    # devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)


    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl