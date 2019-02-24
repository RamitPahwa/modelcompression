import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import data_loader
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


# For CIFAR-10
# cifar-10 name-class map
name_class={'airplane':0,'automobile':1,'bird':2,'cat':3,'deer':4,'dog':5,'frog':6,'horse':7,'ship':8,'truck':9}
name = ['dog','cat','deer','horse']
name_cifar10_vehicles = ['airplane','automobile','truck']
name_exp1 = ['dog','cat']
# TO-DO Random experiment


# change the names in the CIFARSel functional call

def loader(path, batch_size=200, num_workers=4, pin_memory=True):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        data_loader.CIFARSel(root = path,names = name_cifar10_vehicles ,name_class=name_class,train=True,
                             transform=transforms.Compose([
                       transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

def test_loader(path, batch_size=200, num_workers=4, pin_memory=True):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        data_loader.CIFARSel(root = path,names = name_cifar10_vehicles, name_class=name_class,train=False,
                             transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)

'''
# for CIFAR100
insect_name = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'] 
fruit_name = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']
#cifar-100 name-class map 
meta = unpickle('../data/cifar-100-python/meta')
name_class_cifar100 = {}
for i,name in enumerate(meta['fine_label_names']):
    name_class_cifar100[name]=i
# uncomment from below 
# change the names in the CIFARSel functional call
def loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        data_loader.CIFAR100Sel(root = path,names = insect_name ,name_class=name_class_cifar100,train=True,
                             transform = transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.RandomSizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

def test_loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        data_loader.CIFAR100Sel(root = path,names = insect_name, name_class=name_class_cifar100,train=False,
                             transform = transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)
'''
print('OK')
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import data_loader
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


# For CIFAR-10
# cifar-10 name-class map
name_class={'airplane':0,'automobile':1,'bird':2,'cat':3,'deer':4,'dog':5,'frog':6,'horse':7,'ship':8,'truck':9}
name = ['dog','cat','deer','horse']
name_cifar10_vehicles = ['airplane','automobile','truck']
name_exp1 = ['dog','cat']
# TO-DO Random experiment


# change the names in the CIFARSel functional call

def loader(path, batch_size=200, num_workers=4, pin_memory=True):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        data_loader.CIFARSel(root = path,names = name_cifar10_vehicles ,name_class=name_class,train=True,
                             transform=transforms.Compose([
                       transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

def test_loader(path, batch_size=200, num_workers=4, pin_memory=True):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        data_loader.CIFARSel(root = path,names = name_cifar10_vehicles, name_class=name_class,train=False,
                             transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)

'''
# for CIFAR100
insect_name = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'] 
fruit_name = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']
#cifar-100 name-class map 
meta = unpickle('../data/cifar-100-python/meta')
name_class_cifar100 = {}
for i,name in enumerate(meta['fine_label_names']):
    name_class_cifar100[name]=i
# uncomment from below 
# change the names in the CIFARSel functional call
def loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        data_loader.CIFAR100Sel(root = path,names = insect_name ,name_class=name_class_cifar100,train=True,
                             transform=transforms.Compose([
                       transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

def test_loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        data_loader.CIFAR100Sel(root = path,names = insect_name, name_class=name_class_cifar100,train=False,
                             transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)
'''
print('OK')
