# pointnet-tensorflow2.0+

pointnet@tensorflow2.0+

### Introduction and thesis

https://stanford.edu/~rqi/pointnet/

### Thesis pdf download

https://arxiv.org/pdf/1612.00593.pdf

### Reference Code

https://keras.io/examples/vision/pointnet/

## Model structure in the paper

![pointnet_paper](https://github.com/iubizi/pointnet-tensorflow2.0/blob/main/pointnet_paper.png)

## Model structure draw by pydot with high dpi(in pointnet.py)

![pointnet_pydot](https://github.com/iubizi/pointnet-tensorflow2.0/blob/main/pointnet_pydot.png)

I have written a simple and easy-to-use program that does not require a Linux environment (such as commands like cp), and can be adapted to win10 or win11.

Run as: python pointnet.py

First, you need to download a dataset such as ModelNet10 or ModelNet40, and then use ModelNet10_build_dataset_in_npz.py or ModelNet40_build_dataset_in_npz.py in dataset_building_tools to construct a compressed npz format dataset (reading a single file is very slow, using fit_generator will affect code efficiency), Then use read_npz_test.py to read the npz file to see if the npz database is properly constructed.

This program adds earlystopping and visualization of loss and accuracy during training, and reduces the learning rate, so that the training process can be clearly seen.

This program requires 16GB of memory and 8GB of video memory. The pretrained model runs on a 5950x CPU and a 3090Ti GPU. The running time is less than 20 minutes. The model saved in this code only contains weights. Because the model cannot be serialized, it is impossible to save the model.

The pretrained model is the best performing model out of 20 training runs.

### [ModelNet10] loss and acc data for each epoch in training

![ModelNet10](https://github.com/iubizi/pointnet-tensorflow2.0/blob/main/ModelNet10_result/visualization%40ModelNet10.png)

### [ModelNet40] loss and acc data for each epoch in training

![ModelNet40](https://github.com/iubizi/pointnet-tensorflow2.0/blob/main/ModelNet40_result/visualization%40ModelNet40.png)
