from keras.datasets import cifar10
from keras.datasets import cifar100
import numpy as np 
from autokeras import ImageClassifier

# loadning cifar10 from keras

(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# cifar-10 name-class map SELECT the Desired Subset
name_class={'airplane':0,'automobile':1,'bird':2,'cat':3,'deer':4,'dog':5,'frog':6,'horse':7,'ship':8,'truck':9}
name_cifar10_animals = ['dog','cat','deer','horse']
name_cifar10_vehicles = ['airplane','automobile','truck']
name_exp1 = ['dog','cat']

# select the time you want to run the search for in hrs 
time_array =[1,2,3,4,5,6,7,8,9,10]
# change index to change time for NAS
time =  time_array[0]

X_train_selected = []
y_train_selected = []
for i in range(len(X_train)):
    for name in name_cifar10_animals:
        if y_train[i][0] == name_class[name]:
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
  for name in name_cifar10_animals:
    if y_test[i][0] == name_class[name]:
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


clf = ImageClassifier(verbose=True, augment=True, searcher_args={'trainer_args':{'max_iter_num':7}})
clf.fit(X_train_selected, y_train_selected, time_limit=(1*60*60))

clf.final_fit(X_train_selected, y_train_selected, X_test_selected, y_test_selected, retrain=False)
