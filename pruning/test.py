import os
import torch
import argparse
import warnings
import numpy as np
import data_loader
import dataset
import time 
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

def numParams(model):
        return sum([len(w.view(-1)) for w in model.parameters()])

def test(model, test_loader):
    model.eval()
    global best_accuracy
    correct = 0
    for idx, (data, target) in enumerate(test_loader):
        # if cuda:
        #     data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        # do the forward pass
        score = model.forward(data)
        pred = score.data.max(1)[1] # got the indices of the maximum, match them
        correct += pred.eq(target.data).cpu().sum()
    num_params = numParams(model)
    print("predicted {} out of {}".format(correct, len(test_loader.dataset)))
    val_accuracy = correct / float(len(test_loader.dataset)) * 100.0
    print("accuracy = {:.2f}".format(val_accuracy))

    # now save the model if it has better accuracy than the best model seen so forward
    return val_accuracy/100.0, numParams

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, default='model')
    parser.add_argument("--testpath", type = str, default='../data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    # print(torch.cuda.is_available())
    # if torch.cuda.is_available():
    #     model = torch.load(args.model).cuda()
    # else :
    model = torch.load(args.model)
    test_path = args.testpath 
    test_loader = dataset.test_loader(test_path)
    start_time = time.time()
    accuracy, num_params = test(model, test_loader)
    inference_time = time.time() - start_time
    print('Accuracy:'+str(accuracy))
    print('num_params:'+str(num_params))
    print('inference_time:'+str(inference_time))
    

