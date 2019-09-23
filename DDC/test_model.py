import os
import torch
import argparse
import warnings
from utils import numParams
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('model', type=str,
                    help='Path to model')
parser.add_argument('teachermodel', type=str,
                    help='Path to teacher model')

parser.add_argument('dataset', type=str, choices=['mnist', 'cifar10', 'cifar10_old', 'cifar100', 'svhn', 'caltech256'],
                    help='Name of dataset')
parser.add_argument('--new', type=bool, default=False, help='HELP')
parser.add_argument('--file_name',type=str,help='Name of output file')
args = parser.parse_args()
import time
# ----DATASETS----
if args.dataset == 'mnist':
    import datasets.mnist as dataset
elif args.dataset == 'cifar10':
    import datasets.cifar10 as dataset
elif args.dataset == 'cifar10_old':
    import datasets.cifar10_old as dataset
elif args.dataset == 'cifar100':
    import datasets.cifar100 as dataset
elif args.dataset == 'svhn':
    import datasets.svhn as dataset
elif args.dataset == 'caltech256':
    import datasets.caltech256 as dataset
elif args.dataset == 'imagenet':
    import datasets.imagenet as dataset
else:
    print('Dataset not found: ' + args.dataset)
    quit()


if args.new:
    orig_params = numParams(torch.load(args.teachermodel))
    with open(args.file_name+'.out', 'w+') as ofile:
        print('open')
        for filename in os.listdir(args.model):
            if filename.endswith('.net'):
                model = torch.load(os.path.join(args.model, filename))
                dataset.net = model.cuda()

                start_time = time.time()
                acc = dataset.test()
                run_time = time.time() - start_time
                print(run_time)
                ofile.write('---------------------' + filename + '-----------------------\n')
                ofile.write('Accuracy = ' + str(acc) + '\n')
                ofile.write('Time = ' + str(run_time) + '\n')
                ofile.write('Num Params = ' + str(numParams(model)) + '\n')
                ofile.write('Ratio = ' + str(numParams(model)/orig_params) + '\n')

else:
    model = torch.load(args.model)
    dataset.net = model.cuda()
    start = time.time()
    acc = dataset.test()
    print(time.time()-start)
# acc= dataset.train()
