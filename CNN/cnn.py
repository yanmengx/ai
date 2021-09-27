#!/usr/bin/env python

import numpy as np
import tensorflow as tf

# 下载mnist数据集，一共55000张28*28的图片
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，
# 它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
# 一般来说第一个维度是指 batch_size，而 batch_size 一般不限制死的，在训练的时候可能是几十个一起训练，
# 但是在运行模型的时候就一个一个预测，这时候 batch_size 就为 1 .
input_x = tf.placeholder(tf.float32, [None, 28*28])
output_y = tf.placeholder(tf.int32, [None, 10])

# 输入层 28*28*1，灰度图的通道为1
# tf.reshape(tensor,shape,name=None)将tensor变换为参数shape形式
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])

# 从测试集中取3000张图片用来测试
test_x = mnist.test.images[: 3000]
# 3000张图片对应的标签
test_y = mnist.test.labels[: 3000]

conv1 = tf.layers.conv2d(
    inputs=input_x_images,
    filters=32,
    kernel_size=[5, 5],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)
print(conv1)

pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[2, 2],
    strides=2
)
print(pool1)

conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)

pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2, 2],
    strides=2
)

flat = tf.reshape(pool2, [-1, 7*7*64])

dense = tf.layers.dense(
    inputs=flat,
    units=1024,
    activation=tf.nn.relu
)
print(dense)

dropout = tf.layers.dropout(
    inputs=dense,
    rate=0.5
)
print(dropout)

logits = tf.layers.dense(
    inputs=dropout,
    units=10
)
print(logits)

loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)
