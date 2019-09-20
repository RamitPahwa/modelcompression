import torch
import copy 
from Layer import * 
from utils import *
from architecture import *
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch import autograd
import numpy as np 
from config import *

try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    str = str
    unicode = str
    bytes = bytes
    basestring = (str,bytes)
else:
    # 'unicode' exists, must be Python 2
    str = str
    unicode = unicode
    bytes = str
    basestring = basestring



class Controller:
    '''
    This Class defines the Controller Class which 
    '''
    def __init__(self, controllerClass, input_size, output_size, hidden_size, num_layers, lr=0.003, skipSupport=False, kwargs={}):
        self.input_size = input_size
        self.output_size = output_size 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kwargs = kwargs

        if isinstance(controllerClass, basestring):
            print('succcess')
            self.controller = torch.load(controllerClass)
        else:
            self.controller = controllerClass(input_size, output_size, hidden_size, num_layers, **kwargs)
        self.optimizer = optim.Adam(self.controller.parameters(), lr=lr)
        self.skipSupport = skipSupport
        self.actionSeqs = []

    def update_controller(self, avgR, b):
        for actions in self.actionSeqs:                          
            if isinstance(actions, list):
                for action in actions:                           
                    action.reinforce(avgR - b)                   
            else:                                                
                actions.reinforce(avgR - b)                      
            self.optimizer.zero_grad()                           
            autograd.backward(actions, [None for _ in actions])  
            self.optimizer.step()                                
        self.actionSeqs = []                                     

    def rolloutActions(self, layers):
        num_input  = self.input_size
        num_hidden = self.hidden_size
        num_layers = self.num_layers
        num_directions = 2 if (('bidirectional' in self.kwargs) and (self.kwargs['bidirectional'])) else 1
        hn = Variable(torch.zeros(num_layers * num_directions, 1, num_hidden))
        cn = Variable(torch.zeros(num_layers * num_directions, 1, num_hidden))
        input = Variable(torch.Tensor(len(layers), 1, num_input))
        for i in range(len(layers)):
            input[i] = Layer(layers[i]).toTorchTensor(skipSupport=self.skipSupport)
        actions = self.controller(input, (hn, cn))
        self.actionSeqs.append(actions)
        return actions

def accuracy_func2(acc, baseline_acc, threshold1 = 0.60, threshold2 = 0.65):
    '''
    This is updated accuracy reward function wrt accuracy incorporating two threshold, different defination of reward at different threshold
    '''
    R = acc/baseline_acc
    R_thres_1 = threshold1/baseline_acc 
    R_thres_2 = threshold2/baseline_acc

    if R < R_thres_1:
        return 2.7285463*R*R*R-0.035366
    elif R_thres_1 < R < R_thres_2:
        return -1.214 + 2.94683*R
    else:
        return np.tanh(-2.9+5.8*R)

def accuracy_func3(acc, baseline_acc, threshold = 0.65):
    '''
    This the final reward transformation wt accuracy, essentially a shifted tanh transformation
    Input : 
        acc Accurracy of the new compressed model 
        baseline_acc Accuracy of the original model

    Returns:
        Transformed Accuracy
    '''
    R = acc/baseline_acc
    R_threshold = threshold/baseline_acc
    return 0.5+0.5*np.tanh(10*(R-R_threshold))

def accuracy_func(acc, baseline_acc, threshold = 0.65 ):
    '''
    Exponential transformation for accuracy reward 
    Input : 
        acc Accurracy of the new compressed model 
        baseline_acc Accuracy of the original model

    Returns:
        Transformed Accuracy 
    '''
    R = acc/baseline_acc
    R_thres = threshold/baseline_acc
    if R < R_thres:
        return R/2
    else :
        return np.log((1+np.exp(R-R_thres))/2)

def inference_time_func(run_time, threshold=0.60):
    '''
        Input : Inference time defined as run_time
        return:
            transformed Inference time
    '''
    return 1.0/(1.0 + np.exp((run_time-threshold)*10))

def compression(comp_ratio,threshold=0.80):
    return (1.0-1.0/(1.0 + np.exp((comp_ratio-threshold)*10)))

def getEpsilon(iter, max_iter=15.0):
    return min(1, max(0, (1-iter/float(max_iter))**4)) #return 0
'''
def getConstrainedReward(R_a, R_c, cons, vars, iter):
    eps = getEpsilon(iter)
    modelSize = vars[0]
    modelSizeConstraint = cons[0]
    if modelSize > modelSizeConstraint:
        return (eps - 1) + eps * (R_a * R_c)
    else:
        return R_a * R_c
'''
def accuracy_new(acc,base_acc,threshold=Accuracy_threshold):
    acc_ratio=acc/base_acc
    if FUNC=='tanh':
        print("In Tanh")
        return 0.5+0.5*np.tanh(10*(acc_ratio-threshold))
    print("Accuracy threshold is {}".format(threshold))
    # acc_ratio=acc/base_acc
    return (1.0-1.0/(1.0 + np.exp((acc_ratio-threshold)*15)))

def inference_new(run_time, threshold=Inference_threshold):
    '''
        Input : Inference time defined as run_time
        return:
            transformed Inference time
    '''
    return 1.0/(1.0 + np.exp((run_time-threshold)*15))

def compression_new(comp_ratio, threshold=Compression_threshold):
    '''
        Input : Inference time defined as run_time
        return:
            transformed Inference time
    '''
    return 1.0/(1.0 + np.exp((comp_ratio-threshold)*15))

def getConstrainedReward(R_a, R_c, acc, params, it, acc_constraint, size_constraint, epoch, soft=True):
    '''
    This function for hard contraints on accuracy as well as size 
    Input:
        R_a : Reward for accuracy 
        R_c : Reward for compression
        it: Reward for inference time
        acc_contraint : Accuracy constraint 
        Size_contraint: Size contraint 
    '''
    eps = getEpsilon(epoch) if soft else 0
    if (size_constraint and params > size_constraint) or (acc_constraint and acc < acc_constraint):
        return (eps - 1) + eps * (R_a)*(R_c)*1.0/(it)
    return (R_a)*(R_c)*1.0/(it)


def Reward(acc, params, baseline_acc, baseline_params,run_time, size_constraint=None, acc_constraint=None, epoch=-1):
    '''
    Compute reward Combination of Accuracy, Compression, and Inference time
    '''
    R_a = accuracy_new(acc/baseline_acc, Accuracy_threshold)
    R_c = compression_new(params/baseline_params, Compression_threshold)
    R_t = inference_new(run_time, Inference_threshold)
    Total_Reward = R_a * R_c * R_t
    print("Total Reward:"+str(Total_Reward))
    return Total_Reward
    # R_a = acc/baseline_acc
    # C = (float(baseline_params - params))/baseline_params
    # R_c = C*(2-C)
    # return (R_a) * (R_c)
    # print("In reward")
    # R_a = accuracy_func3(acc,baseline_acc) #if acc > 0.92 else -1
    # # R_a = acc/baseline_acc
    # it = inference_time_func(run_time)
    # # print("Runtime : {}".format(run_time))
    # C = (float(baseline_params - params))/baseline_params
    # # R_c transformation as defined in the paper
    # R_c = C*(2-C)
    # return (R_a)*(R_c)*it


previousModels = {}
def rollout_batch(model, controller, architecture, dataset, N, e,parent_runtime, acc_constraint=None, size_constraint=None):
    '''
    This generates a model given a parent model by removing layers to produce new models if the new models return is None then a reward of -1 is assigned 
    Input:
        Model : Parent Model 
        Controller : Compression Policy LSTM or AutoRegressive LSTM
        Architecture : Use the Architecture function called in run.py class at Architecture.py
        dataset: called in run.py
        N: 

    '''
    model_statistics= [] 
    model_reward = [-1]*N
    model_time = [-1]*N
    model_accuracy = [-1]*N
    model_compression = [-1]*N
    newModels = []
    idxs = []
    Rs = [0]*N
    actionSeqs = []
    studentModels = []
    for i in range(N):
        model_ = copy.deepcopy(model)
        layers = layersFromModule(model_)
        actions = controller.rolloutActions(layers)
        actionSeqs.append(actions)
        newModel = architecture.generateChildModel([a.data.numpy()[0] for a in actions])
        hashcode = hash(str(newModel)) if newModel else 0
        if hashcode in previousModels and constrained == False:
            Rs[i] = previousModels[hashcode]
        elif newModel is None:
            Rs[i] = -1
        else:
            # print(newModel)
            #torch.save(newModel, modelSavePath + '%f_%f.net' % (e, i))
            newModels.append(newModel)
            studentModels.append(newModel)
            idxs.append(i)
    accs = []
    # accs = trainNormalParallel(studentModels, dataset, epochs=5) if architecture.datasetName is 'caltech256' else trainTeacherStudentParallel(model, studentModels, dataset, epochs=5)
    run_time = []
    try :
        accs, run_time = trainStudentTeacherParallelNew(model, studentModels, dataset, epochs=STUDENT_EPOCHES)
        # accs, run_time = trainTeacherStudentParallel(model, studentModels, dataset, epochs=STUDENT_EPOCHES)
        for i in range(len(idxs)):
            model_accuracy[idxs[i]] = accs[i]
            model_time[idxs[i]] = run_time[i]
        print (accs)
        print (run_time)
    except ValueError:
        print("Value something ")
    for acc in accs:
        print('Val accuracy: %f' % acc)
    for i in range(len(newModels)):
        model_compression[i] = 1.0 - (float(numParams(newModels[i]))/architecture.parentSize)
        print('Compression: %f' % (1.0 - (float(numParams(newModels[i]))/architecture.parentSize)))
    
    # time_ratio = run_time/parent_runtime
    #R = [Reward(accs[i], numParams(newModels[i]), architecture.baseline_acc, architecture.parentSize, iter=int(e), constrained=constrained, vars=[numParams(newModels[i])], cons=[1700000]) for i in range(len(accs))]
    print(acc_constraint,"constrained")
    print("parent_runtime : {}".format(parent_runtime))
    R = [Reward(accs[i], numParams(newModels[i]), architecture.baseline_acc, architecture.parentSize, run_time[i]/parent_runtime , size_constraint=size_constraint, acc_constraint=acc_constraint, epoch=e) for i in range(len(accs))]
    for i in range(len(idxs)):
        Rs[idxs[i]] = R[i]
    for i in range(len(Rs)):
        print('Reward achieved %f' % Rs[i])
    model_reward = Rs
    model_statistics.extend([model_reward,model_accuracy,model_compression,model_time])
    # input()
    return (Rs, actionSeqs, newModels,model_statistics)


def rollouts(N, model, controller, architecture, dataset, e,parent_runtime, size_constraint=None, acc_constraint=None):
    '''
    Calls rollout batch function
    return reward , sequence of action (keep a layer or not ) , and compressed models
    '''
    Rs = []
    actionSeqs = []
    models = []
    (Rs, actionSeqs, models,model_statistics) = rollout_batch(copy.deepcopy(model), controller, architecture, dataset, N, e,parent_runtime, acc_constraint=acc_constraint, size_constraint=size_constraint)
    return (Rs, actionSeqs, models,model_statistics)

