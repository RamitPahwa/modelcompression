#!/usr/bin/env python
import re
import os
import sys
from copy import deepcopy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def accuracy_func(acc, baseline_acc=0.92, threshold = 0.35 ):
    R = acc/baseline_acc
    R_thres = threshold/baseline_acc
    if R < R_thres:
        return R/2
    else :
        return np.log((1+np.exp(R-R_thres))/2)

def inference_time_func(run_time, threshold=0.6):
    return 1.0/(1.0 + np.exp((run_time/3.0-threshold)*10))

regex = '-*(\d*)\.0*_(\d)\.0*\.net-*' 
#print(sys.argv[1])
with open(sys.argv[1], 'r') as f:
    data = f.read()
    data = data.split('\n')
    #data.pop()
    #data.pop()
    iters = [x for x in data if '----------' in x]
    accuracy = [x for x in data if 'Accuracy' in x]
    ratio = [x for x in data if 'Ratio' in x]
    #ratio.append('Ratio = ' + str(0.4486))
    time = [x for x in data if 'Time' in x]

    iters = [re.search(regex, x) for x in iters]
    iters = [(int(x.group(1)), int(x.group(2))) for x in iters]
    #print(iters)
    accuracy = [float(x.split(' ')[2]) for x in accuracy]
    ratio = [float(x.split(' ')[2]) for x in ratio]
    time = [float(x.split(' ')[2]) for x in time]

    iter_x = list(zip(*iters))
    d = list(zip(iter_x[0], iter_x[1], accuracy, ratio, time))
    print(len(d))
    #print(d)
    tmps = []
    msx_tmps = {}
    new_d = []
    for x in d:
        if str(x[0]) not in msx_tmps.keys():
            tmps.append(x[0])
            msx_tmps[str(x[0])] = x
            print(len(tmps))
            print(x, msx_tmps[str(x[0])])
            print('------------------')
            #new_d.append(x)
        elif msx_tmps[str(x[0])][3] < x[3]:
            print(x, msx_tmps[str(x[0])])
            msx_tmps[str(x[0])] = x
            #new_d.append(x)

    fig = plt.figure()
    d = list(zip(*d))
    print(len(d[0]))
    plt.scatter(iter_x[0], ratio)
    plt.savefig(sys.argv[2] + 'same.png')
    plt.close()

    print(msx_tmps['31'])
    print(len(tmps))
    for k, v in msx_tmps.items():
        new_d.append(v)

    new_d = list(zip(*new_d))
    
    fig = plt.figure()
    plt.ylim(ymax=1.0)
    plt.ylim(ymin=0.0)
    plt.scatter(d[0], d[2], c='r')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(accuracy, ratio, time)
    #ax.set_xlabel('Accuracy')
    #ax.set_ylabel('Ratio')
    #ax.set_zlabel('Time')
    plt.savefig(sys.argv[2] + 'acc.png')
    plt.close()

    fig2 = plt.figure()
    plt.ylim(ymax=1.0)
    plt.ylim(ymin=0.0)
    plt.xlabel('Iteration')
    plt.ylabel('Compression')
    plt.scatter(d[0],d[3], c='b')
    plt.savefig(sys.argv[2] + 'comp.png')
    plt.close()
    '''
    mod_acc = [accuracy_func(x) for x in accuracy]
    mod_time = [inference_time_func(x) for x in time] 
    mod_ratio = [(1-x)*(1+x) for x in ratio]
    mod_acc = np.asarray(mod_acc)
    mod_time = np.asarray(mod_time)
    mod_ratio = np.asarray(mod_ratio)
    
    reward = mod_acc*mod_time*mod_ratio
    fig = plt.figure()
    plt.scatter(iter_x[0], mod_acc)
    plt.savefig(sys.argv[2] + 'rew_acc.png')
    plt.ylim(ymax=0.24)
    plt.close()
    fig = plt.figure()
    plt.scatter(iter_x[0], mod_time)
    plt.savefig(sys.argv[2] + 'rew_time.png')
    plt.close()
    fig = plt.figure()
    plt.scatter(iter_x[0], mod_ratio)
    plt.savefig(sys.argv[2] + 'rew_ratio.png')
    plt.close()
    fig = plt.figure()
    plt.scatter(iter_x[0], reward)
    plt.savefig(sys.argv[2] + 'reward.png')
    plt.close()

'''