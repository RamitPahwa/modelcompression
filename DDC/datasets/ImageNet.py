import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import datasets, transforms
from torchvision import models
from folder_dataloader import ImageFolderSel

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
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
#traindir = '/home/ubuntu/inapp/modelcompression/imagenet/exp1/train'
traindir = '/code/imagenet/exp1/train'
valdir = '/code/imagenet/exp1/val'

train_loader = torch.utils.data.DataLoader(
        ImageFolderSel(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])),batch_size=batch_size, shuffle=True,**kwargs)
test_loader = torch.utils.data.DataLoader(
        ImageFolderSel(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])),batch_size=batch_size, shuffle=False,**kwargs)
print(len(train_loader.dataset))
print(len(test_loader.dataset))

avg_loss = list()
best_accuracy = 0.0

# train the network
optimizer = None
def train(epoch):
    global optimizer
    if epoch == 1:
        #optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    global avg_loss
    correct = 0
    net.train()
    for b_idx, (data, targets) in enumerate(train_loader):
        if b_idx >= 250000:
            break
        if cuda:
            data, targets = data.cuda(), targets.cuda()
        # convert the data and targets into Variable and cuda form
        data, targets = Variable(data), Variable(targets)

        # train the network
        optimizer.zero_grad()
        scores = net.forward(data)
        loss = F.nll_loss(scores, targets)

        # compute the accuracy
        pred = scores.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(targets.data).cpu().sum()

        avg_loss.append(loss.data[0])
        loss.backward()
        optimizer.step()

        if b_idx % log_schedule == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (b_idx+1) * len(data), len(train_loader.dataset),
                100. * (b_idx+1)*len(data) / len(train_loader.dataset), loss.data[0]))

    # now that the epoch is completed plot the accuracy
    train_accuracy = correct.double() / float(len(train_loader.dataset))
    print("training accuracy ({:.2f}%)".format(100*train_accuracy))
    return (train_accuracy*100.0)


def test():
    net.eval()
    global best_accuracy
    correct = 0
    for idx, (data, target) in enumerate(test_loader):
        if idx >= 100000:
            break
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        score = net.forward(data)
        pred = score.data.max(1)[1] # got the indices of the maximum, match them
        correct += pred.eq(target.data).cpu().sum()

    print("predicted {} out of {}".format(correct, len(test_loader.dataset)))
    val_accuracy = correct.double() / float(len(test_loader.dataset)) * 100.0
    print("accuracy = {:.2f}".format(val_accuracy))

    # now save the model if it has better accuracy than the best model seen so forward
    return val_accuracy/100.0

