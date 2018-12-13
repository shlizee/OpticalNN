# OpticalNN
Based on the model of AlexNet, we proposed a new network called Optical Neural Network (ONN), which replaces the first layer of AlexNet with optical set-up. More details can be found in the paper: "Optical Coprocessor for Artificial Neural Network" which is placed in doc.

## Requirements
- Python
- Tensorflow
- Numpy

## Content
- doc: our paper and papers that we quoted
- alexnet: code for training alexnet, which can be found https://github.com/kratzert/finetune_alexnet_with_tensorflow we mainly use this code for benchmarking accuracy of alexnet and design our network
- benchmark: benchmarking of alexnet with can be found in tensorflow tutorial
- sqnl_alexnet: modified first layer of AlexNet from using ReLU to square non-linearity, also didn't include bias in first layer. We use sqnl_alexnet as a comparision to our ONN network
- onn: our ONN model

## How to use
- To reproduce the paper result of ONN, first run onn_setup.ipynb to set up parameters of the 4f-system, this produces the corresponding xx.npy, yy.npy, Lambda.npy, k_z_values.npy files. you can also modify the optical setup.  Then run onn_train.py. When training, be sure to put train.txt, val.txt, test.txt files in the same directory as training code. Way to write the above three files are as follows:
```
/path/to/txt/file/image1.png 0
/path/to/txt/file/image2.png 1...
```
- To see how our pretrained kernel in ONN differs from AlexNet, run compare_sqnl_and_onn.ipynb
- run other two training files (namely sqnl_alexnet_train.py, finetune.py) to see the performace of SqnlAlexNet and AlexNet
