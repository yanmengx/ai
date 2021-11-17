import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


mnist = input_data.read_data_sets('mnist_data', one_hot=True)


'''
VGG共有5段卷积，每段内有2-3个卷积层，每段卷积层后面接一个最大池化层来缩小图片的
尺寸。段内的卷积核数量一样，越往后卷积核越多，分布如下：64-128-256-512-512。
最后面是3个全连接层。
数据集用的mnist，网络做了一些更改，第一段改为一个卷积层，后面四段两个卷积层，卷积
核数量为16-32-64-128-128。最后3个全连接层：256-256-10.
'''


'''
定义卷积层
input_op: 输入的tensor
name: 该层的名称
kh: 卷积核的高
kw: 卷积核的宽
n_out: 卷积核数量（输出通道）
dh: 步长的高
dw: 步长的宽
p:参数列表
'''
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    # 获取输入图像的通道数
    n_in = input_op.get_shape()[-1].value
    # 定义一个命名空间,命名空间对变量的使用没有任何影响，不会导致在命名空间内可以使用在其他地方就不可以使用的情况
    # （1）在某个tf.name_scope()指定的区域中定义的所有对象及各种操作，他们的“name”属性上会增加该命名区的区域名，用以区别对象属于哪个区域；
    # （2）将不同的对象及操作放在由tf.name_scope()指定的区域中，便于在tensorboard中展示清晰的逻辑关系图，这点在复杂关系图中特别重要
    with tf.name_scope(name) as scope:
        # 随机生成卷积核
        # tf.get_variable(name,
        #                   shape=None,
        #                   dtype=None,
        #                   initializer=None,
        #                   regularizer=None,
        #                   trainable=True,
        #                   collections=None,
        #                   caching_device=None,
        #                   partitioner=None,
        #                   validate_shape=True,
        #                   use_resource=None,
        #                   custom_getter=None)
        # 该函数的作用是创建新的tensorflow变量，常见的initializer有：常量初始化器tf.constant_initializer、
        # 正太分布初始化器tf.random_normal_initializer、截断正态分布初始化器tf.truncated_normal_initializer、
        # 均匀分布初始化器tf.random_uniform_initializer。
        #
        # tf.keras.initializers.glorot_normal(seed=1)
        # Glorot正态分布初始化方法，也称作Xavier正态分布初始化，参数由0均值，标准差为sqrt(2 / (fan_in + fan_out))的正态分布产生，
        # 其中fan_in和fan_out是权重张量的扇入扇出（即输入和输出单元数目），seed：随机数种子
        kernel = tf.get_variable(scope + "w", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.keras.initializers.glorot_normal(seed=1))
        conv = tf.nn.conv2d(input_op, kernel, strides=[1, dh, dw, 1], padding='SAME')
        bias_init_val = tf.constant(0.1, shape=[n_out], dtype=tf.float32)
        # trainable：如果为True，则会默认将变量添加到图形集合GraphKeys.TRAINABLE_VARIABLES中。
        # 此集合用于优化器Optimizer类优化的的默认变量列表[可为optimizer指定其他的变量集合]，可就是要训练的变量列表。
        bias = tf.Variable(bias_init_val, trainable=True, name="b")
        activation = tf.nn.relu(conv+bias, name=scope)

        p += [kernel, bias]
    return activation


'''
定义全连接层
input_op: 输入的tensor
name: 该层的名称
n_out: 输出通道数
p:参数列表
'''
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.keras.initializers.glorot_normal(seed=1))
        bias = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name="b")
        activation = tf.nn.relu(tf.matmul(input_op, kernel)+bias, name=scope)

        p += [kernel, bias]
    return activation


'''
定义最大池化层
input_op: 输入的tensor
name: 该层的名称
kh: 池化的高
kw: 池化的宽
dh: 步长的高
dw: 步长的宽
'''
def max_pool(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1],
                          padding='SAME', name=name)


'''
VGG
'''
input_x = tf.placeholder('float', [None, 28*28])
output_y = tf.placeholder('float', [None, 10])
# dropout的概率
keep_prob = tf.placeholder('float')
input_x_image = tf.reshape(input_x, [-1, 28, 28, 1])

p = []
conv1_1 = conv_op(input_x_image, name='conv1_1', kh=3, kw=3, n_out=16, dh=1, dw=1, p=p)
pool1 = max_pool(conv1_1, name='pool1', kh=2, kw=2, dw=2, dh=2)

conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3, n_out=32, dh=1, dw=1, p=p)
conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=32, dh=1, dw=1, p=p)
pool2 = max_pool(conv2_2, name='pool2', kh=2, kw=2, dw=2, dh=2)

conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
pool3 = max_pool(conv3_2, name='pool3', kh=2, kw=2, dh=2, dw=2)

conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
pool4 = max_pool(conv4_2, name='pool4', kh=2, kw=2, dh=2, dw=2)

conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
pool5 = max_pool(conv5_2, name='pool5', kh=2, kw=2, dh=2, dw=2)

pool5_shape = pool5.get_shape()
# 把图像铺平的另一种方式，shape的第一个是图片的数量，第二个第三个是图像的长宽，第四个是同一图片不同卷积核的输出，也可以理解为通道
flattened_shape = pool5_shape[1].value * pool5_shape[2].value * pool5_shape[3].value
resh1 = tf.reshape(pool5, [-1, flattened_shape], name='resh1')

fc6 = fc_op(resh1, name='fc6', n_out=256, p=p)
fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')

fc7 = fc_op(fc6_drop, name='fc7', n_out=256, p=p)
fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc7_drop')

fc8 = fc_op(fc7_drop, name='fc8', n_out=10, p=p)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=output_y, logits=fc8)
)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
accuracy_op = tf.metrics.accuracy(
    labels=tf.argmax(output_y, axis=1),
    predictions=tf.argmax(fc8, axis=1)
)[1]

# 开始训练和测试
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
# saver.save(sess, 'model')

test_output = sess.run(fc8, {input_x: mnist.test.images[10: 50], keep_prob: 1.0})
inferenced_y = np.argmax(test_output, 1)
print('Inferenced numbers:', inferenced_y)
print('Real numbers: ', np.argmax(mnist.test.labels[10: 50], 1))

sess.close()


