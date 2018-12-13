# OpticalNN
Based on the model of AlexNet (code written in tensorflow at https://github.com/kratzert/finetune_alexnet_with_tensorflow), we proposed a new network called Optical Neural Network (ONN), which replaces the first layer of AlexNet with optical set-up. More details can be found in the paper: "Optical Coprocessor for Artificial Neural Network" which is in arxiv.

## Requirements
- Python
- Tensorflow
- Numpy

## Content
- benchmark: benchmarking of alexnet code with can be found in tensorflow tutorial
- sqnl_alexnet: modified first layer of AlexNet from using ReLU to square non-linearity, also didn't include bias in first layer. We use sqnl_alexnet as a comparision to our ONN network
- onn: our ONN model

## How to use
- To reproduce the paper result of ONN, first run onn_setup.ipynb to set up parameters of the 4f-system, this generates the corresponding xx.npy, yy.npy, Lambda.npy, k_z_values.npy files. you can also modify the optical setup.  Then run onn_train.py. When training, be sure to put train.txt, val.txt, test.txt files in the same directory as training code. Way to write the above three files are as follows:
```
/path/to/txt/file/image1.png 0
/path/to/txt/file/image2.png 1...
```
- To see difference between AlexNet convolution (using AlexNet pretrained kernel, can be found in https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/) and ONN convolution (using ONN trained kernel, can be found in onn directory), run compare_sqnl_and_onn.ipynb
- run sqnl_alexnet_train.py to see the performace of SqnlAlexNet.
