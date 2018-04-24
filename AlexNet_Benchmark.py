import tensorflow as tf
import numpy as np
import cv2
from scipy.misc import imread
from numpy import *
import time

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, name, padding="SAME", group=1):
    
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    with tf.variable_scope(name) as scope:
        
        if group==1:
            conv = convolve(input, kernel)
        else:
            input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
            kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
relu = tf.nn.relu(bias, name=scope.name)

return relu

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)

def get_slice(kernel, i, gpu_num):
    i=tf.constant(gpu_num)
    slices=kernel.shape[3]//gpu_num
    dim1=kernel.shape[0]
    dim2=kernel.shape[1]
    dim3=kernel.shape[2]
    size = [dim1,dim2,dim3,slices]
    start=[dim1,dim2,dim3,i*slices]
    return tf.slice(kernel,start,size)

def get_slice2(b,i,gpu_num):
    i=tf.constant(gpu_num)
    slices=b.shape[0]//gpu_num
    start=i*slices
    return tf.slice(b,[start],[slices])

x = tf.placeholder(tf.float32, [1, 227, 227, 3])
kernel = tf.placeholder(tf.float32, [11, 11, 3, 96])
b=tf.placeholder(tf.float32,[96])
kernel2 = tf.placeholder(tf.float32, [5, 5, 96, 256])
b2=tf.placeholder(tf.float32,[256])
kernel3 = tf.placeholder(tf.float32, [3, 3, 128, 384])
b3=tf.placeholder(tf.float32,[384])
kernel4=tf.placeholder(tf.float32, [3, 3, 192, 384])
b4=tf.placeholder(tf.float32,[384])
kernel5=tf.placeholder(tf.float32, [3, 3, 192, 256])
b5=tf.placeholder(tf.float32,[256])
ks=[]
bs=[]
convs=[]
pools=[]
norms=[]

t_sum_conv1=[]
t_sum_norm1=[]
t_sum_pool1=[]
sum_conv1_1=0
sum_norm1_1=0
sum_pool1_1=0
sum_conv1_2=0
sum_norm1_2=0
sum_pool1_2=0
for j in range(10):
    for i in range(2):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:
                #                 1
                slice_n=get_slice(kernel,i,2)
                b_1=get_slice2(b,i,2)
                t1=time.time()
                conv1 = conv(x, slice_n, b_1, 11,11, 48, 4, 4, name='conv1')
                t2=time.time()
                norm=lrn(conv1, 2, 1e-05, 0.75, name='norm1')
                t3=time.time()
                #                     print("norm1:",time.time()-t)
                pool=max_pool(norm, 3, 3, 2, 2, padding='VALID', name='pool1')
                    t4=time.time()
                    #                     print("pool1:",time.time()-t)
                    ks.append(slice_n)
                    bs.append(b_1)
                    convs.append(conv1)
                    norms.append(norm)
                    pools.append(pool)
                    t_sum_conv1.append(t2-t1)
                    t_sum_norm1.append(t3-t2)
                    t_sum_pool1.append(t4-t3)



with tf.device('/cpu:0'):
    pool1=tf.concat([pools[0],pools[1]],axis=3)
#     print("2")
#                 2

t_sum_conv2=[]
t_sum_conv3=[]
t_sum_conv4=[]
t_sum_conv5=[]
t_sum_norm2=[]
t_sum_pool2=[]
t_sum_pool3=[]
sum_conv2_1=0
sum_conv3_1=0
sum_conv4_1=0
sum_conv5_1=0
sum_norm2_1=0
sum_pool2_1=0
sum_pool3_1=0
sum_conv2_2=0
sum_conv3_2=0
sum_conv4_2=0
sum_conv5_2=0
sum_norm2_2=0
sum_pool2_2=0
sum_pool3_2=0
for j in range(10):
    for i in range(2):
        with tf.device('/gpu:0'):
            with tf.name_scope('tower_%d' % i) as scope:
                #                 print("3")
                slice_n2=get_slice(kernel2,i,2)
                b_2=get_slice2(b2,i,2)
                t5=time.time()
                conv2=conv(pool1, slice_n2, b_2, 5, 5, 128, 1, 1, name='conv2')
                #                     print("conv2:",time.time()-t)
                t6=time.time()
                    norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
                    #                     print("norm2:",time.time()-t)
                    t7=time.time()
                    pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
                    t8=time.time()
                    #                     print("pool2:",time.time()-t)
                    
                    #                 3
                    slice_n3=get_slice(kernel3,i,2)
                    b_3=get_slice2(b3,i,2)
                    t9=time.time()
                    conv3 = conv(pool2, slice_n3 ,b_3, 3, 3, 192, 1, 1, name='conv3')
                    t10=time.time()
                    #                     print("conv3:",time.time()-t)
                    # #                 4
                    slice_n4=get_slice(kernel4,i,2)
                    b_4=get_slice2(b4,i,2)
                    t11=time.time()
                    conv4 = conv(conv3, slice_n4, b_4 ,3, 3, 192, 1, 1, name='conv4')
                    t12=time.time()
                    #                     print("conv4:",time.time()-t)
                    #                 5
                    slice_n5=get_slice(kernel5,i,2)
                    b_5=get_slice2(b5,i,2)
                    t13=time.time()
                    conv5 = conv(conv4, slice_n5, b_5 ,3, 3, 128, 1, 1, name='conv4')
                    #                     print("conv5:",time.time()-t)
                    t14=time.time()
                    pool3 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
                    t15=time.time()
                    #                     print("pool3:",time.time()-t)
                    t_sum_conv2.append(t6-t5)
                    t_sum_norm2.append(t7-t6)
                    t_sum_pool2.append(t8-t7)
                    t_sum_conv3.append(t10-t9)
                    t_sum_conv4.append(t12-t11)
                    t_sum_conv5.append(t14-t13)
                    t_sum_pool3.append(t15-t14)
                    ks.append([(slice_n2),(slice_n3),(slice_n4),(slice_n5)])
                    bs.append([(b_2),(b_3),(b_4),(b_5)])
                    convs.append([(conv2),(conv3),(conv4),(conv5)])
                    norms.append(norm2)
                    pools.append([(pool2),(pool3)])
#                 print("4")

for i in range (20):
    if i % 2 ==0:
        sum_conv1_1=sum_conv1_1+t_sum_conv1[i]
        sum_norm1_1=sum_norm1_1+t_sum_norm1[i]
        sum_pool1_1=sum_pool1_1+t_sum_pool1[i]
        sum_conv2_1=sum_conv2_1+t_sum_conv2[i]
        sum_norm2_1=sum_norm2_1+t_sum_norm2[i]
        sum_pool2_1=sum_pool2_1+t_sum_pool2[i]
        sum_conv3_1=sum_conv3_1+t_sum_conv3[i]
        sum_conv4_1=sum_conv4_1+t_sum_conv4[i]
        sum_conv5_1=sum_conv5_1+t_sum_conv5[i]
        sum_pool3_1=sum_pool3_1+t_sum_pool3[i]
    if (i+1) %2 ==0:
        sum_conv1_2=sum_conv1_2+t_sum_conv1[i]
        sum_norm1_2=sum_norm1_2+t_sum_norm1[i]
        sum_pool1_2=sum_pool1_2+t_sum_pool1[i]
        sum_conv2_2=sum_conv2_2+t_sum_conv2[i]
        sum_norm2_2=sum_norm2_2+t_sum_norm2[i]
        sum_pool2_2=sum_pool2_2+t_sum_pool2[i]
        sum_conv3_2=sum_conv3_2+t_sum_conv3[i]
        sum_conv4_2=sum_conv4_2+t_sum_conv4[i]
        sum_conv5_2=sum_conv5_2+t_sum_conv5[i]
        sum_pool3_2=sum_pool3_2+t_sum_pool3[i]



