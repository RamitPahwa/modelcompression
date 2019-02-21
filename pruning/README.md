
Python 2.7.13
torch = 1.0.0

Download data using `wget` inside `modelcompression/data` create data inside modelcompression directory
`wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar`
`wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tagz`
`wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz`
Caltech 256 loader is commented in `data_loader.py`

This demonstrates pruning a VGG16 based classifier that classifies a small dog/cat dataset.
This was able to reduce the CPU runtime by x3 and the model size by x4.
For more details you can read the [blog post](https://jacobgil.github.io/deeplearning/pruning-deep-learning).

At each pruning step 512 filters are removed from the network.


Usage
-----
In `dataset.py` make change according to dataset either CIFAR10 or CIFAR100 as well as the subset we want and comment the other loader.Use `time` in front of python command for prunning time

In `finetune.py` change the output classes according to the no. of classes in subset in `ModifiedVGG16Model` `ModifiedVGG11Model` and `ModifiedVGG19Model`

Training:
`python finetune.py --train --train_path ../data/ --test_path ../data/ --dataset CIFAR10 --arch VGG16 --subset animals`

Pruning:
`python finetune.py --prune --train_path ../data/ --test_path ../data/ --dataset CIFAR10 --arch VGG16 --subset animals`

values to pass 
--dataset CIFAR10 or CIFAR100 or CAL256
--arch VGG16/VGG19/VGG11
--subset animals.vehicles/insect/fruits depending upon the ones passed in `dataset.py`.(argument used just for naming )

Testing 

`python test.py --model [MODEL_PATH] --testpath ../data/`

[MODEL_PATH]: PATH to original or prunned model ( saves entire model not state_dict())
The dataset loader is same as used in trainin and pruning and you will need to uncomment the required portion in `dataset.py`
