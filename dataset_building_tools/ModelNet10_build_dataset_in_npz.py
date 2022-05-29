import gc
gc.enable()

####################
# 读取off文件
####################

import glob
import os
import trimesh

import numpy as np

x_train, y_train = [], []
x_test, y_test = [], []

class_map = {}

folders = glob.glob('ModelNet10/*')

for i, folder in enumerate(folders):
    
    print('processing class: {}'.format(os.path.basename(folder)))
    # store folder name with ID so we can retrieve later
    class_map[i] = folder.split('/')[-1]
    # gather all files
    train_files = glob.glob(os.path.join(folder, 'train/*'))
    test_files = glob.glob(os.path.join(folder, 'test/*'))
    
    for f in train_files:
        x_train.append(trimesh.load(f).sample(2048))
        y_train.append(i)

    for f in test_files:
        x_test.append(trimesh.load(f).sample(2048))
        y_test.append(i)

####################
# 数据处理和可视化
####################

from tensorflow.keras.utils import to_categorical
    
x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = to_categorical(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)
y_test = to_categorical(y_test)

print()
print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)

print('x_test.shape =', x_test.shape)
print('y_test.shape =', y_test.shape)
print()

print('class_map =', class_map)

####################
# 存储为npz压缩格式
####################

np.savez_compressed( 'ModelNet10.npz',
          
                     x_train = x_train,
                     y_train = y_train,
          
                     x_test = x_test,
                     y_test = y_test )
