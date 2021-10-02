#!/usr/bin/env python

import numpy as np
# import tensorflow as tf
# tensorflow2.0不支持placeholder，要改成这个样子才能用placeholder
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# 下载mnist数据集，一共55000张28*28的图片
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# 模型：输入层->卷积层->池化层->卷积层->池化层->全连接层->全连接层->输出层

'''
tf.placeholder(
    dtype, 数据类型。常用的是tf.float32,tf.float64等数值类型
    shape=None, 数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
    name=None 名称
)
Tensorflow的设计理念称之为计算流图，在编写程序时，首先构筑整个系统的graph，代码并不会直接生效.
graph为静态的，然后，在实际的运行时，启动一个session，程序才会真正的运行
placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，
它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
'''

# 一般来说第一个维度是指 batch_size，而 batch_size 一般不限制死的，在训练的时候可能是几十个一起训练，
# 但是在运行模型的时候就一个一个预测，这时候 batch_size 就为 1 .
# none意思是tensor 的第一维度可以是任意维度
input_x = tf.placeholder(tf.float32, [None, 28*28])
# 输出是一个one-hot向量
output_y = tf.placeholder(tf.int32, [None, 10])

# 输入层 28*28*1，灰度图的通道为1
'''
tf.reshape(
    tensor, Tensor张量
    shape, Tensor张量，用于定义输出张量的shape，组成元素类型为 int32或int64
    name=None 可选参数，用于定义操作名称
)
将tensor变换为参数shape形式，不会更改张量中元素的顺序或总数
'''
# -1代表图像的个数，因为不知道个数，就用-1，python会自动算出来
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])

# 从测试集中取3000张图片用来测试
test_x = mnist.test.images[: 3000]
# 3000张图片对应的标签
test_y = mnist.test.labels[: 3000]

'''
tf.layers.conv2d 参数：
inputs: 输入，张量
filters：卷积核个数，也就是卷积层的厚度
kernel_size：卷积核的尺寸
strides：扫描步长
padding：边上补0，valid不需要补0，same需要补0
activation：激活函数
'''
# conv1 5*5*32
# 单张图片输入时是28*28*1，输出是28*28*32
conv1 = tf.layers.conv2d(
    inputs=input_x_images,
    filters=32,
    kernel_size=[5, 5],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)
print(conv1)

'''
tf.layers.max_pooling2d 参数：
inputs：输入，张量必须要有4个维度
pool_size：过滤器的尺寸
'''
# 池化后维度减半
# 输入时是28*28*32，输出时是14*14*32
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

# 平坦化，变成一维，后面进入全连接层
flat = tf.reshape(pool2, [-1, 7*7*64])

'''
tf.layers.dense 参数：
inputs：张量
units：神经元的个数
activation：激活函数 
'''
# 全连接层 1024
dense = tf.layers.dense(
    inputs=flat,
    units=1024,
    activation=tf.nn.relu
)
print(dense)

'''
tf.layers.dropout：参数
inputs：张量
rate：丢弃率
training：是否在训练的时候丢弃
'''
dropout = tf.layers.dropout(
    inputs=dense,
    rate=0.5
)
print(dropout)

# 输出层，本质上是是一个全连接层，但不用激活函数
# 输出[,10]
logits = tf.layers.dense(
    inputs=dropout,
    units=10
)
print(logits)

'''
tf.losses.softmax_cross_entropy 参数：
onehot_labels：标签值
logits：神经网络的输出
'''
# 计算误差（cross_entropy交叉熵），在用softmax计算百分比的概率
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)
print(loss)
# 用Adam优化器来最小化误差，学习率为0.001，类似梯度下降
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

'''
tf.metrics.accuracy 参数：
labels：真实标签
predictions：预测值
return：（accuracy，update_op）accuracy是一个张量准确率，update_op是一个op可以求出精度

tf.argmax 参数：
input, 输入矩阵
axis=None, 
name=None,
dimension=None,
output_type=tf.int64
用途：返回最大的那个数值所在的下标（第一个参数是矩阵，第二个参数是0或者1。0表示的是按列比较返回
最大值的索引，1表示按行比较返回最大值的索引）。
'''
# 精度。计算预测值和实际标签的匹配程度
accuracy_op = tf.metrics.accuracy(
    labels=tf.argmax(output_y, axis=1),
    predictions=tf.argmax(logits, axis=1)
)[1]

# 创建会话
sess = tf.Session()
# 初始化变量，全局和局部
# group吧很多操作弄成一个组
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())
sess.run(init)

for i in range(20000):
    batch = mnist.train.next_batch(50)
    train_loss, _ = sess.run([loss, train_op], {input_x: batch[0], output_y: batch[1]})
    if i % 100 == 0:
        test_accuracy = sess.run(accuracy_op, {input_x: test_x, output_y: test_y})
        print("Step=%d, Train loss=%.4f, [Test accuracy=%.2f]" % (i, train_loss, test_accuracy))

test_output = sess.run(logits, {input_x: test_x[: 20]})
inferenced_y = np.argmax(test_output, 1)
print(inferenced_y, 'Inferenced numbers')
print(np.argmax(test_y[: 20], 1), 'Real numbers')

sess.close()
