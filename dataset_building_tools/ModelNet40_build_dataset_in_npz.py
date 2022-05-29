import gc
gc.enable()

####################
# 读取h5文件
####################

import h5py
import numpy as np
 
f = h5py.File('modelnet40_ply_hdf5_2048/ply_data_train0.h5', 'r')

x_train = np.array(f['data'])
y_train = np.array(f['label'])

f.close()

for name in ['1', '2', '3', '4']:

    f = h5py.File('modelnet40_ply_hdf5_2048/ply_data_train'+name+'.h5', 'r')

    x_train = np.concatenate((x_train, np.array(f['data'])))
    y_train = np.concatenate((y_train, np.array(f['label'])))

    f.close()

# print(x_train.shape)
# print(y_train.shape)

f = h5py.File('modelnet40_ply_hdf5_2048/ply_data_test0.h5', 'r')

x_test = np.array(f['data'])
y_test = np.array(f['label'])

f.close()

f = h5py.File('modelnet40_ply_hdf5_2048/ply_data_test1.h5', 'r')

x_test = np.concatenate((x_test, np.array(f['data'])))
y_test = np.concatenate((y_test, np.array(f['label'])))

f.close()

# print(x_test.shape)
# print(y_test.shape)



####################
# 数据处理和可视化
####################

from tensorflow.keras.utils import to_categorical
    
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)

print('x_test.shape =', x_test.shape)
print('y_test.shape =', y_test.shape)
print()



####################
# 存储为npz压缩格式
####################

np.savez_compressed( 'ModelNet40.npz',
          
                     x_train = x_train,
                     y_train = y_train,
          
                     x_test = x_test,
                     y_test = y_test )
