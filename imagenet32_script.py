import pickle
import numpy as np
import os

base_path = "data/imagenet32"


def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo)
    return dictionary


temp1 = unpickle(os.path.join(base_path, 'train_data_batch_1'))
temp2 = unpickle(os.path.join(base_path, 'train_data_batch_2'))
temp3 = unpickle(os.path.join(base_path, 'train_data_batch_3'))
temp4 = unpickle(os.path.join(base_path, 'train_data_batch_4'))
temp5 = unpickle(os.path.join(base_path, 'train_data_batch_5'))
temp6 = unpickle(os.path.join(base_path, 'train_data_batch_6'))
temp7 = unpickle(os.path.join(base_path, 'train_data_batch_7'))
temp8 = unpickle(os.path.join(base_path, 'train_data_batch_8'))
temp9 = unpickle(os.path.join(base_path, 'train_data_batch_9'))
temp10 = unpickle(os.path.join(base_path, 'train_data_batch_10'))

data_tuple = (temp1['data'], temp2['data'], temp3['data'], temp4['data'], temp5['data'],
              temp6['data'], temp7['data'], temp8['data'], temp9['data'], temp10['data'])

label_tuple = (temp1['labels'], temp2['labels'], temp3['labels'], temp4['labels'], temp5['labels'],
               temp6['labels'], temp7['labels'], temp8['labels'], temp9['labels'], temp10['labels'])

temp_data = np.concatenate(data_tuple, axis=0)
temp_labels = np.concatenate(label_tuple, axis=0)
np.savez(os.path.join(base_path, 'imagenet32_train'), data=temp_data, labels=temp_labels)

loaded = np.load(os.path.join(base_path, 'imagenet32_train'+'.npz'))
print(loaded.files)
print(loaded['data'].shape)
print(loaded['labels'].shape)

temp_file = unpickle(os.path.join(base_path, 'val_data'))
np.savez(os.path.join(base_path, 'imagenet32_val'), data=temp_file['data'], labels=temp_file['labels'])

loaded = np.load(os.path.join(base_path, 'imagenet32_val'+'.npz'))
print(loaded.files)
print(loaded['data'].shape)
print(loaded['labels'].shape)
