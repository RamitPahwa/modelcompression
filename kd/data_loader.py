from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data

class CIFARSel(data.Dataset):
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    @property
    def targets(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels

    def __init__(self, root,names,name_class,train = True,transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform  # training set or test set
        self.train  = train
        if self.train:
            self.train_data = []
            self.train_labels = []
            self.train_data_selected = []
            self.train_labels_selected =[]

            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            # select the provided classes 
            for i in range(len(self.train_data)):
                for name in names:
                    if self.train_labels[i] == name_class[name]:
                        self.train_data_selected.append(self.train_data[i])
                        self.train_labels_selected.append(self.train_labels[i])
            '''
            self.labels_ids = set(self.train_labels_selected)  

            self.labels_id_dict = {}
            counter = 0 
            for i in self.labels_ids:
                self.labels_id_dict[i] = counter 
                counter = counter + 1
            for i in range(len(self.train_labels_selected)):
                self.train_labels_selected[i] = self.labels_id_dict[self.train_labels_selected[i]]          
            '''  
            self.train_labels_selected = self.train_labels_selected[:int(1.0*len(self.train_labels_selected))]
            self.train_data_selected = self.train_data_selected[:int(1.0*len(self.train_data_selected))]
        else:
            self.test_data_selected =[]
            self.test_labels_selected =[]

            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
                
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

            for i in range(len(self.test_data)):
                for name in names:
                    if self.test_labels[i] == name_class[name]:
                        self.test_data_selected.append(self.test_data[i])
                        self.test_labels_selected.append(self.test_labels[i])
            '''
            self.labels_ids = set(self.test_labels_selected)
            self.labels_id_dict = {}
            counter = 0 
            for i in self.labels_ids:
                self.labels_id_dict[i] = counter 
                counter = counter + 1

            for i in range(len(self.test_labels_selected)):
                self.test_labels_selected[i]= self.labels_id_dict[self.test_labels_selected[i]]
            '''
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data_selected[index], self.train_labels_selected[index]
        else:
            img, target = self.test_data_selected[index], self.test_labels_selected[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data_selected)
        else:
            return len(self.test_data_selected)

#name_class={'airplane':0,'automobile':1,'bird':2,'cat':3,'deer':4,'dog':5,'frog':6,'horse':7,'ship':8,'truck':9}
#dataset =  CIFARSel('data/',['airplane','cat','truck','deer'],name_class=name_class)
    

class CIFAR100Sel(CIFARSel):
    'Inherits CIFARSel Class'

    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# dataset =  CIFAR100Sel('/code/N2N',['beetle'],name_class=name_class)
# print(len(dataset))
'''
def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img.resize((224, 224))

class Caltech256(data.Dataset):
    def __init__(self, split='train', path='../data/caltech256/256_ObjectCategories'):
        super(Caltech256, self).__init__()
        self.path = path
        self.split = split
        self.filepaths = glob(join(self.path, '*/*.jpg'))
        n = len(self.filepaths)
        train_paths, test_paths = self.get_splits(self.path, 1001)
        if split == "train":
            self.filepaths = train_paths#list(map(lambda i: self.filepaths[i], train_paths))
        else:
            #test_choices = filter(lambda i: i not in train_choices, range(len(self.filepaths)))
            self.filepaths = test_paths#list(map(lambda i: self.filepaths[i], test_paths))
        self.targets = [f.split('/')[-1] for f in glob(join(self.path, '*'))]
    
    def get_splits(self, base_path, seed=1000):
        np.random.seed(seed)
        train_files = []
        test_files = []
        # From each class select 10% at random
        classes = [f.split('/')[-1] for f in glob(join(base_path, '*'))]
        for c in classes:
            files = glob(join(base_path, c, '*'))
            n = len(files)
            #train = np.random.choice(files, int(n*0.8), replace=False)
            train = np.random.choice(files, n - 15, replace=False)
            test = filter(lambda x: x not in train, files)
            train_files.extend(train)
            test_files.extend(test)
        return train_files, test_files
    
    def __getitem__(self, index):
        filepath = self.filepaths[index]
        img = img_transform(load_img(filepath))
        # Scale and convert to tensor
        target = torch.Tensor([self.targets.index(filepath.split('/')[-2])])
        return img, target
    
    def __len__(self):
        return len(self.filepaths)
'''