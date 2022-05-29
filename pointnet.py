import gc
gc.enable()

####################
# 引入库
####################

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from matplotlib import pyplot as plt
plt.rcParams['savefig.dpi'] = 200 # 图片像素

####################
# 避免占满
####################

gpus = tf.config.experimental.list_physical_devices('GPU')

for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

####################
# 读取测试
####################

doc = np.load('ModelNet40.npz')
N_CLASS = 40 # 和数据类型数量一致

x_train = doc['x_train']
y_train = doc['y_train']

x_test = doc['x_test']
y_test = doc['y_test']

####################
# 模型代码块
####################

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding='valid')(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation('relu')(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation('relu')(x)


class OrthogonalRegularizer(keras.regularizers.Regularizer):
  
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))


def tnet(inputs, num_features):
    
    # 将偏差初始化为单位矩阵
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer='zeros',
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # 将仿射变换应用于输入特征
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

####################
# 网络
####################

inputs = keras.Input(shape=(2048, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(N_CLASS, activation='softmax')(x)

model = keras.Model( inputs = inputs,
                     outputs = outputs,
                     name = 'pointnet' )

# 打印模型
# model.summary()

# 保存成流程图片
'''
keras.utils.plot_model( model,
                        to_file = 'pointnet.png',
                        show_shapes = True,
                        show_layer_names = True,
                        dpi = 200 )
'''

####################
# 编译模型
####################

model.compile( loss = 'categorical_crossentropy',
               optimizer = keras.optimizers.Adam(learning_rate=5e-4), # 0.001
               metrics = ['accuracy'] )

####################
# 图表
####################

class PlotProgress(keras.callbacks.Callback):

    def __init__(self, entity = ['loss', 'accuracy']):
        self.entity = entity

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        
        self.losses = []
        # self.val_losses = []

        self.accs = []
        self.val_accs = []

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        # 损失函数
        self.losses.append(logs.get('{}'.format(self.entity[0])))
        # self.val_losses.append(logs.get('val_{}'.format(self.entity[0])))
        # 准确率
        self.accs.append(logs.get('{}'.format(self.entity[1])))
        self.val_accs.append(logs.get('val_{}'.format(self.entity[1])))

        self.i += 1
        
        plt.figure( figsize = (6, 3) )

        plt.subplot(121)
        plt.plot(self.x, self.losses, label="{}".format(self.entity[0]))
        # plt.plot(self.x, self.val_losses, label="val_{}".format(self.entity[0]))
        plt.legend()
        plt.title('loss')
        plt.grid()

        plt.subplot(122)
        plt.plot(self.x, self.accs, label="{}".format(self.entity[1]))
        plt.plot(self.x, self.val_accs, label="val_{}".format(self.entity[1]))
        plt.legend()
        plt.title('accuracy')
        plt.grid()

        plt.tight_layout() # 减少白边
        plt.savefig('visualization@ModelNet40.png')
        plt.close() # 关闭

####################
# 回调函数
####################

# 绘图函数
plot_progress = PlotProgress(entity = ['loss', 'accuracy'])

# 早产
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping( monitor = 'val_accuracy',
                                patience = 20,
                                restore_best_weights = True )

####################
# 训练
####################

model.fit( x_train, y_train,
           validation_data = (x_test, y_test),
           
           epochs = 10000, batch_size = 32,

           callbacks = [plot_progress, early_stopping],

           # max_queue_size = 16,
           workers = 8, # 多进程核心数
           use_multiprocessing = True, # 多进程

           shuffle = True, # 再次打乱
           verbose = 1, # 2 一次一行 1 动态进度条
           )

####################
# 保存模型
####################

# 只保存验证集准确率最高的那个
model.save_weights('pointnet_weights@ModelNet40.h5')
