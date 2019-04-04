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
from datasets import dataset

json_path='./params.json'
parser = argparse.ArgumentParser(description='N2N: Network to Network Compression using Policy Gradient Reinforcement Learning')

parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar10_old', 'cifar100', 'svhn', 'caltech256'],
                    help='Name of dataset')
parser.add_argument('--teacherModel', type=str,
                    help='Path to teacher model')
parser.add_argument('--model', type=str, required=False,
                    help='Path to base model architecture if different from teacherModel')
parser.add_argument('--cuda', type=bool, required=False, default=True,
                    help='Use GPU or not')
parser.add_argument('--gpuids', type=list, required=False, default=[0],
                    help='Which GPUs to use')
parser.add_argument('--debug', type=bool, required=False, default=False,
                    help='Debug mode')
parser.add_argument('--size_constraint', type=int, required=False,
                    help='Add a constraint on size in # parameters')
parser.add_argument('--acc_constraint', type=float, required=False,
                    help='Add a constraint on accuracy in [0, 1]')
parser.add_argument('--controller', type=str, required=False,
                    help='Path to a previously trained controller')
parser.add_argument('--model_save_path',type=str,required=True,help='Path to save model checkpoints')
args = parser.parse_args()


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


if len(args.gpuids) > 1:
    print('Parallel version not implemented yet')
else:
    torch.cuda.set_device(int(args.gpuids[0]))

# ----DATASETS----
if args.dataset == 'mnist':
    import datasets.mnist as dataseta
elif args.dataset == 'cifar10':
    import datasets.cifar10 as dataseta
elif args.dataset == 'cifar10_old':
    import datasets.cifar10_old as dataseta
elif args.dataset == 'cifar100':
    import datasets.cifar100 as dataseta
elif args.dataset == 'svhn':
    import datasets.svhn as dataseta
elif args.dataset == 'caltech256':
    import datasets.caltech256 as dataseta
elif args.dataset == 'imagenet':
    import datasets.imagenet as dataseta
else:
    print('Dataset not found: ' + args.dataset)
    quit()

print('Using %s as dataset' % args.dataset)
dataset.cuda = args.cuda
# print(dataset.test_loader.dataset[0])
datasetInputTensor = dataseta.test_loader.dataset[0][0].unsqueeze(0)
print(datasetInputTensor.size())

print("loading model")
teacherModel = torch.load(args.teacherModel)
print("model loaded")

# # Identify baseline accuracy of base model
# dataset.net = model.cuda() if args.cuda else model
# print('Testing parent model to determine baseline accuracy')
# import time
# startTime = time.time()
# baseline_acc = baseline_acc if baseline_acc != None else dataset.test()
# parent_runtime = time.time() - startTime

params = Params(json_path)

    # use GPU if available
params.cuda = torch.cuda.is_available()
# model = net.Net(params).cuda() if params.cuda else net.Net(params)
dmodel = dnet.NET2('net').cuda() if params.cuda else dnet.NET2('net')

accs, run_time = trainTeacherStudent(teacherModel, dmodel, dataset, epochs=20)
student_size=numParams(dmodel)
parent_size=numParams(teacherModel)

compression=1.0-(float(student_size)/float(parent_size))
print("Accuracy ",accs)
print("inference ",run_time)
print("compression ",compression)
print("parent_size ",parent_size)
print("student_size ",student_size)
modelSavePath = './'+ args.model_save_path
if not os.path.exists(modelSavePath):
    os.mkdir(modelSavePath)
resultsFile = open(os.path.join(modelSavePath, 'results.txt'), "w")
resultsFile.write("accuracy " + str(accs) + '\n')
resultsFile.write("Inference time " + str(run_time) + '\n')
resultsFile.write("compression " + str(compression) + '\n')
resultsFile.write("parent_size " + str(parent_size) + '\n')
resultsFile.write("student_size " + str(student_size) + '\n')
