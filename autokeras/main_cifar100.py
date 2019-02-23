from keras.datasets import cifar10
from keras.datasets import cifar100
import numpy as np 
from autokeras import ImageClassifier

# loadning cifar10 from keras

(X_train, y_train), (X_test, y_test) = cifar100.load_data()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
# for CIFAR100
insect_name = ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'] 
fruit_name = ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper']
#cifar-100 name-class map 
meta = unpickle('../data/cifar-100-python/meta')
name_class_cifar100 = {}
for i,name in enumerate(meta['fine_label_names']):
    name_class_cifar100[name]=i

# select the time you want to run the search for in hrs 
time_array =[1,2,3,4,5,6,7,8,9,10]
# change index to change time for NAS
time =  time_array[0]

X_train_selected = []
y_train_selected = []
for i in range(len(X_train)):
    for name in insect_name:
        if y_train[i][0] == name_class_cifar100[name]:
            X_train_selected.append(X_train[i])
            y_train_selected.append(y_train[i])

X_train_selected = np.array(X_train_selected)
y_train_selected = np.array(y_train_selected)
y_train_selected_ravel = y_train_selected.ravel()
print(y_train_selected_ravel[0])
print(y_train_selected[0][0])
labels_ids = set(y_train_selected_ravel)  

labels_id_dict = {}
counter = 0 
for i in labels_ids:
    labels_id_dict[i] = counter 
    counter = counter + 1
for i in range(len(y_train_selected)):
    y_train_selected[i] = labels_id_dict[y_train_selected_ravel[i]]
#   return X_train_selected, y_train_selected


# X_train_selected, y_train_selected =out(name,name_class)
y_train_selected = y_train_selected[:int(1.0*len(y_train_selected))]
X_train_selected = X_train_selected[:int(1.0*len(X_train_selected))]
print(X_train_selected.shape)
print(y_train_selected.shape)

X_test_selected = []
y_test_selected = []
for i in range(len(X_test)):
  for name in insect_name:
    if y_test[i][0] == name_class_cifar100[name]:
      X_test_selected.append(X_test[i])
      y_test_selected.append(y_test[i])
X_test_selected = np.array(X_test_selected)
y_test_selected = np.array(y_test_selected)
y_test_selected_ravel = y_test_selected.ravel()
# print(y_test_selected_ravel[0])
# print(y_test_selected.shape)
labels_ids = set(y_test_selected_ravel)  

labels_id_dict = {}
counter = 0 
for i in labels_ids:
    labels_id_dict[i] = counter 
    counter = counter + 1

for i in range(len(y_test_selected)):
    y_test_selected[i]= labels_id_dict[y_test_selected_ravel[i]]
print(X_train_selected.shape)
print(y_train_selected.shape)
print(X_test_selected.shape)
print(y_test_selected.shape)
clf = ImageClassifier(verbose=True, augment=True, searcher_args={'trainer_args':{'max_iter_num':7}})
clf.fit(X_train_selected, y_train_selected, time_limit=(1*60*60))

clf.final_fit(X_train_selected, y_train_selected, X_test_selected, y_test_selected, retrain=False)
