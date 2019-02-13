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
class ModifiedVGG16Model(torch.nn.Module):
	def __init__(self):
		super(ModifiedVGG16Model, self).__init__()

		model = models.vgg16(pretrained=True)
		self.features = model.features

		for param in self.features.parameters():
			param.requires_grad = False

		self.classifier = nn.Sequential(
		    nn.Dropout(),
		    nn.Linear(25088, 4096),
		    nn.ReLU(inplace=True),
		    nn.Dropout(),
		    nn.Linear(4096, 4096),
		    nn.ReLU(inplace=True),
		    nn.Linear(4096, 5))

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

class ModifiedVGG19Model(torch.nn.Module):
	def __init__(self):
		super(ModifiedVGG19Model, self).__init__()

		model = models.vgg19(pretrained=True)
		self.features = model.features

		for param in self.features.parameters():
			param.requires_grad = False

		self.classifier = nn.Sequential(
		    nn.Dropout(),
		    nn.Linear(25088, 4096),
		    nn.ReLU(inplace=True),
		    nn.Dropout(),
		    nn.Linear(4096, 4096),
		    nn.ReLU(inplace=True),
		    nn.Linear(4096, 4))

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

class ModifiedVGG11Model(torch.nn.Module):
	def __init__(self):
		super(ModifiedVGG11Model, self).__init__()

		model = models.vgg11(pretrained=True)
		self.features = model.features

		for param in self.features.parameters():
			param.requires_grad = False

		self.classifier = nn.Sequential(
		    nn.Dropout(),
		    nn.Linear(25088, 4096),
		    nn.ReLU(inplace=True),
		    nn.Dropout(),
		    nn.Linear(4096, 4096),
		    nn.ReLU(inplace=True),
		    nn.Linear(4096, 4))

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

def numParams(model):
        return sum([len(w.view(-1)) for w in model.parameters()])

def test(model, test_loader):
    model.eval()
    global best_accuracy
    correct = 0
    for idx, (data, target) in enumerate(test_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
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
    parser.add_argument("--arch", type=str, default="VGG16")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.arch == "VGG16": 
        if torch.cuda.is_available():
            model = ModifiedVGG16Model().cuda()
        else:
            model = ModifiedVGG16Model()
    elif args.arch == "VGG19":
        if torch.cuda.is_available():
            model = ModifiedVGG19Model().cuda()
        else:
            model = ModifiedVGG19Model()
    elif args.arch == "VGG11":
        if torch.cuda.is_available():
            model = ModifiedVGG11Model().cuda()
        else:
            model = ModifiedVGG11Model()

    model.load_state_dict(torch.load(args.model))
    
    print(torch.cuda.is_available())
    
    test_path = args.testpath 
    test_loader = dataset.test_loader(test_path)
    start_time = time.time()
    accuracy, num_params = test(model, test_loader)
    inference_time = time.time() - start_time
    print('Accuracy:'+str(accuracy))
    print('num_params:'+str(num_params))
    print('inference_time:'+str(inference_time))
    

