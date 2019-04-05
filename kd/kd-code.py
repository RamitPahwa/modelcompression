import os
import torch
from torch import nn 
from torch import optim
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
import argparse
import json

import model.net as net
import model.net2 as dnet

from utils import *
import datasets.cifar10 as dataseta
import datasets.cifar100 as datasetb
import model.resnet 

datasetInputTensor = dataseta.test_loader.dataset[0][0].unsqueeze(0)
print(datasetInputTensor.size())

print("loading model")
teacherModel = torch.load('resnet18_cifar10.net')
print("model loaded")

dmodel = dnet.NET2('net').cuda()

from torchvision import datasets, transforms
from torchvision import models

from torch.utils.data.sampler import SubsetRandomSampler
# from datasets.cifar_dataloader import CIFARSel
from datasets.cifar_dataloader import CIFARSel
batch_size = 200
lr = 1e-3
seed = 1
log_schedule = 10
cuda = True

torch.manual_seed(seed)
if cuda:
    print('Using cuda')
    torch.cuda.manual_seed(seed)
    #torch.cuda.set_device(1)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = datasets.CIFAR10(root='./', train=True,download=True, transform=transforms.Compose([
                       transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

devset = datasets.CIFAR10(root='./', train=False,download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))

trainset_size = len(trainset)
indices = list(range(trainset_size))
split = int(np.floor(0.5 * trainset_size))
np.random.seed(230)
np.random.shuffle(indices)

train_sampler = SubsetRandomSampler(indices[:split])

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,sampler=train_sampler)

test_loader = torch.utils.data.DataLoader(devset, batch_size=batch_size,shuffle=False)

def KDTRAIN(teacher, student, train_loader, epochs=5, lr=0.0005):
    startTime = time.time()
    student = student.cuda()
    teacher = teacher.cuda()
    # If there is a log softmax somewhere, delete it in both teacher and student
    removeLayers(teacher, type='LogSoftmax')
    removeLayers(teacher, type='Softmax')
    removeLayers(student, type='LogSoftmax')
    removeLayers(student, type='Softmax')
    MSEloss = nn.MSELoss().cuda()
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    student.train()
    for i in range(1, epochs+1):
        for b_idx, (data, targets) in enumerate(train_loader):
            data = data.cuda()
            data = Variable(data)
            optimizer.zero_grad()
            studentOutput = student(data)
            teacherOutput = teacher(data).detach()
            loss = MSEloss(studentOutput, teacherOutput)
            loss.backward()
            optimizer.step()
        student.add_module('LogSoftmax', nn.LogSoftmax())
        dataseta.net = student
        removeLayers(student, type='LogSoftmax')
        print(dataseta.test())
        print('Train Epoch: {} \tLoss: {:.6f}'.format(i, loss.data[0]))
    student.add_module('LogSoftmax', nn.LogSoftmax())
    dataseta.net = student
    s2 = time.time()
    acc = dataseta.test()
    run_time = time.time() - s2
    print('Time elapsed: {}'.format(time.time()-startTime))
    return acc ,run_time

accs, run_time = KDTRAIN(teacherModel, dmodel, train_loader, epochs=20)
student_size=numParams(dmodel)
parent_size=numParams(teacherModel)

compression=1.0-(float(student_size)/float(parent_size))
print("Accuracy ",accs)
print("inference ",run_time)
print("compression ",compression)
print("parent_size ",parent_size)
print("student_size ",student_size)
modelSavePath = './'
if not os.path.exists(modelSavePath):
    os.mkdir(modelSavePath)
resultsFile = open(os.path.join(modelSavePath, 'results.txt'), "w")
resultsFile.write("accuracy " + str(accs) + '\n')
resultsFile.write("Inference time " + str(run_time) + '\n')
resultsFile.write("compression " + str(compression) + '\n')
resultsFile.write("parent_size " + str(parent_size) + '\n')
resultsFile.write("student_size " + str(student_size) + '\n')