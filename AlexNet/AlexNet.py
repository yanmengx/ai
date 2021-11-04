import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


mnist = input_data.read_data_sets('mnist_data', one_hot=True)


# 输入是227*227
# 原模型结构：卷积层conv1(卷积-->ReLU-->池化-->归一化)->卷积层conv2(卷积-->ReLU-->池化-->归一化)
#        ->卷积层conv3(卷积-->ReLU)->卷积层conv4(卷积-->ReLU)->卷积层conv5(卷积-->ReLU-->池化)
#        ->全连接层FC6(全连接-->ReLU-->Dropout)->全连接层FC7(全连接-->ReLU-->Dropout)->输出层
# 输入是28*28
# 用在mnist上，网络结构改变了：卷积层conv1(卷积-->ReLU-->池化-->归一化)->卷积层conv2(卷积-->ReLU-->池化-->归一化)
#                       ->卷积层conv3(卷积-->ReLU)->全连接层FC6(全连接-->ReLU-->Dropout)
#                       ->全连接层FC7(全连接-->ReLU-->Dropout)->输出层

# tf.truncated_normal(shape, mean, stddev)：截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成。
# shape，生成张量的维度
# mean，均值
# stddev，标准差
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)：创建数值常量
# value:是一个必须的值，可以是一个数值，也可以是一个列表；可以是一维的，也可以是多维的。
# dtype：数据类型，一般可以是tf.float32, tf.float64等。
# shape：表示张量的“形状”，即维数以及每一维的大小
# name: 可以是任何内容，只要是字符串就行
# verify_shape：默认为False，如果修改为True的话表示检查value的形状与shape是否相符，如果不符会报错
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# tf.nn.conv2d (input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
# input : 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_width, in_channel ]，其中batch为图片的数量，
#         in_height 为图片高度，in_width 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。
# filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_width, in_channel, out_channels ]，
#          其中 filter_height 为卷积核高度，filter_width 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，
#          out_channel 是卷积核数量。
# strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
# padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。
#          "SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
#          设 输入图片 A（image）的大小是MxM，卷积核 K（filter）的大小是：NxN，
#          S：步长   P：输入图片边界填充0（原图片：5x5, 填充后：7x7，P=1）
#          padding=‘valid’
#          out_convshape_height(width) = (M-N) / S +1
#          padding=‘same’
#          out_convshape_height(width) = (M-N+2*P) / S +1
# use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# tf.nn.max_pool(value, ksize, strides, padding, name=None)：最大值池化
# value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，shape是[batch, height, width, channels]
# ksize：池化窗口的大小，取一个四维向量，一般是[batch, height, width, channels]
# strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
# padding：和卷积类似，可以取'VALID' 或者'SAME'，‘SAME’ 为补零，‘VALID’ 则不补
# 返回一个Tensor，类型不变，shape是[batch, height, width, channels]
def max_pool_3x3(x):
    # 池化层：3×3步长为2的池化单元（重叠池化，步长小于池化单元的宽度）
    # H_out=floor((H_in+2*padding-kernerl_size)/stride+1)
    # W_out=floor((W_in+2*padding-kernerl_size)/stride+1)
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


# 局部响应归一化,对局部神经元创建了竞争的机制,使得其中响应小打的值变得更大,并抑制反馈较小的.
# 在2015年 Very Deep Convolutional Networks for Large-Scale Image Recognition.提到LRN基本没什么用。
# tf.nn.lrn(input,depth_radius=None,bias=None,alpha=None,beta=None,name=None)
# 就是一个点沿着channel方向(前 depth_radius + 后depth_radius)的点的平方加，乘以alpha，
# 即 sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
def norm(x, lsize=4):
    return tf.nn.lrn(x, lsize, bias=1.0, alpha=0.001/9.0, beta=0.75)


input_x = tf.placeholder('float', [None, 28*28])
output_y = tf.placeholder('float', [None, 10])
# dropout的概率
keep_prob = tf.placeholder('float')
input_x_image = tf.reshape(input_x, [-1, 28, 28, 1])

w_conv1 = weight_variable([3, 3, 1, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(input_x_image, w_conv1)+b_conv1)
h_pool1 = max_pool_3x3(h_conv1)
h_norm1 = norm(h_pool1, lsize=4)

w_conv2 = weight_variable([3, 3, 64, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_norm1, w_conv2)+b_conv2)
h_pool2 = max_pool_3x3(h_conv2)
h_norm2 = norm(h_pool2, lsize=4)

w_conv3 = weight_variable([3, 3, 128, 256])
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv2d(h_norm2, w_conv3)+b_conv3)
h_pool3 = max_pool_3x3(h_conv3)
h_norm3 = norm(h_pool3, lsize=4)

w_fc1 = weight_variable([4096, 1024])
b_fc1 = bias_variable([1024])
h_norm3_flat = tf.reshape(h_norm3, [-1, 4*4*256])
# tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False,
#           a_is_sparse=False, b_is_sparse=False, name=None)
# a: 一个类型为 float16, float32, float64, int32, complex64, complex128 且张量秩 > 1 的张量。
# b: 一个类型跟张量a相同的张量。
# transpose_a: 如果为真, a则在进行乘法计算前进行转置。
# transpose_b: 如果为真, b则在进行乘法计算前进行转置。
# adjoint_a: 如果为真, a则在进行乘法计算前进行共轭和转置。
# adjoint_b: 如果为真, b则在进行乘法计算前进行共轭和转置。
# a_is_sparse: 如果为真, a会被处理为稀疏矩阵。
# b_is_sparse: 如果为真, b会被处理为稀疏矩阵。
# name: 操作的名字（可选参数）
# 返回值： 一个跟张量a和张量b类型一样的张量且最内部矩阵是a和b中的相应矩阵的乘积。
h_fc1 = tf.nn.relu(tf.matmul(h_norm3_flat, w_fc1)+b_fc1)
# Dropout,抑制过拟合，随机的断开某些神经元的连接或者是不激活某些神经元
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, 256])
b_fc2 = bias_variable([256])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2)+b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

w_fc3 = weight_variable([256, 10])
b_fc3 = bias_variable([10])
logits = tf.matmul(h_fc2_drop, w_fc3)+b_fc3

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=output_y, logits=logits)
)
# 为什么tf.train.AdamOptimizer比tf.train.GradientDescentOptimizer效果好，收敛速度更快刚开始的迭代中adam的准确率远高于gd
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
accuracy_op = tf.metrics.accuracy(
    labels=tf.argmax(output_y, axis=1),
    predictions=tf.argmax(logits, axis=1)
)[1]

sess = tf.Session()

init = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer()
)

sess.run(init)

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy_op, feed_dict={input_x: batch[0], output_y: batch[1], keep_prob: 1.0})
        print('Step {}, training accuracy: {}'.format(i, train_accuracy))
    sess.run(train_step, feed_dict={input_x: batch[0], output_y: batch[1], keep_prob: 0.5})


# saver = tf.train.Saver()
# saver.save(sess, 'model_Adam')

test_output = sess.run(logits, {input_x: mnist.test.images[: 20], keep_prob: 1.0})
inferenced_y = np.argmax(test_output, 1)
print('Inferenced numbers:', inferenced_y)
print('Real numbers: ', np.argmax(mnist.test.labels[: 20], 1))

sess.close()
