import math
import time
import datetime
import tf_slim as slim
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# TF-Slim 是用于定义、训练和评估复杂模型的tensorflow轻量级库。tf-slim的组件能轻易地与原生
# tensorflow框架还有其他的框架（例如tf.contrib.learn）进行整合
# slim = tf.contrib.slim 因为版本的问题已经不能用了
# 产生截断的正态分布
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
parameters = []


'''
定义inception_v3_arg_scope(),
用来生成网络中经常用到的函数的默认参数（卷积的激活函数、权重初始化方式、标准化器等）
weight_decay: 权值衰减系数
stddev: 标准差
batch_norm_var_collection:
'''
def inception_v3_arg_scope(weight_decay=0.00004,
                           stddev=0.1,
                           batch_norm_var_collection='moving_vars'):
    batch_norm_params = {
        'decay': 0.997,  # 衰减系数d
        'epsilon': 0.001,  # 极小值
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection]
        }
    }
    # slim.arg_scope()可以给函数的参数自动赋予某些默认值
    # 例如：
    # slim.arg_scope([slim.conv2d,slim.fully_connected],weights_regularizer=slim.l2_regularizer(weight_decay)):
    # 会对[slim.conv2d,slim.fully_connected]这两个函数的参数自动赋值，
    # 将参数weights_regularizer的默认值设为slim.l2_regularizer(weight_decay)
    # 使用了slim.arg_scope后就不需要每次重复设置参数，只需在修改时设置即可。
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.truncated_normal_initializer(stddev=stddev),  # 权重初始化
        activation_fn=tf.nn.relu,  # 激励函数
        normalizer_fn=slim.batch_norm,  # 标准化器
        normalizer_params=batch_norm_params  # normalizer_params标准化器的参数
    ) as sc:
        # 返回定义好的scope
        return sc
'''
tf.contrib.slim.conv2d(inputs,
          num_outputs,
          kernel_size,
          stride=1,
          padding='SAME',
          data_format=None,
          rate=1,
          activation_fn=nn.relu,
          normalizer_fn=None,
          normalizer_params=None,
          weights_initializer=initializers.xavier_initializer(),
          weights_regularizer=None,
          biases_initializer=init_ops.zeros_initializer(),
          biases_regularizer=None,
          reuse=None,
          variables_collections=None,
          outputs_collections=None,
          trainable=True,
          scope=None)
inputs                        是指需要做卷积的输入图像
num_outputs             指定卷积核的个数（就是filter的个数）
kernel_size               用于指定卷积核的维度（卷积核的宽度，卷积核的高度）
stride                         为卷积时在图像每一维的步长
padding                     为padding的方式选择，VALID或者SAME
data_format              是用于指定输入的input的格式
rate                           对于使用空洞卷积的膨胀率，rate等于1为普通卷积，rate=n代表卷积核中两两数之间插入了n-1个0
activation_fn             用于激活函数的指定，默认的为ReLU函数
normalizer_fn           用于指定正则化函数
normalizer_params  用于指定正则化函数的参数
weights_initializer     用于指定权重的初始化程序
weights_regularizer  为权重可选的正则化程序
biases_initializer       用于指定biase的初始化程序
biases_regularizer    biases可选的正则化程序
reuse                        指定是否共享层或者和变量
variable_collections  指定所有变量的集合列表或者字典
outputs_collections   指定输出被添加的集合
trainable                    卷积层的参数是否可被训练
scope                        共享变量所指的variable_scope
'''


def inception_v3_base(input, scope=None):
    end_points = {}
    '''
    tf.variable_scope(
    name_or_scope,
    default_name=None,
    values=None,
    initializer=None,
    regularizer=None,
    caching_device=None,
    partitioner=None,
    custom_getter=None,
    reuse=None,
    dtype=None,
    use_resource=None,
    constraint=None,
    auxiliary_name_scope=True)
命名域 (name scope)，通过tf.name_scope 或 tf.op_scope创建；
变量域 (variable scope)，通过tf.variable_scope 或 tf.variable_op_scope创建；
这两种作用域，对于使用tf.Variable()方式创建的变量，具有相同的效果，都会在变量名称前面，加上域名称。
对于通过tf.get_variable()方式创建的变量，只有variable scope名称会加到变量名称前面，而name scope不会作为前缀。
    '''
    # 第一部分：基础部分，卷积和池化交错
    with tf.variable_scope(scope, 'inception_V3', [input]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='VALID'):
            net1 = slim.conv2d(input, 32, [3, 3], stride=2, scope='conv2d_1a_3x3')
            net2 = slim.conv2d(net1, 32, [3, 3], scope='conv2d_2a_3x3')
            net3 = slim.conv2d(net2, 64, [3, 3], padding='SAME', scope='conv2d_2b_3x3')
            net4 = slim.max_pool2d(net3, [3, 3], stride=2, scope='maxPool_3a_3x3')
            net5 = slim.conv2d(net4, 80, [1, 1], scope='conv2d_4a_3x3')
            net6 = slim.conv2d(net5, 192, [3, 3], padding='SAME', scope='conv2d_4b_3x3')
            net = slim.max_pool2d(net6, [3, 3], stride=2, scope='maxPool_5a_3x3')
    # 第二部分：Inception模块组：inception_1\inception_2\inception_3
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
        # inceprion_1:第一个模块组，共3个inception_module
        # inception_1_m1:第一组的1号module
        with tf.variable_scope('inception_1_m1'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 64, [1, 1], scope='conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch1_1 = slim.conv2d(net, 48, [1, 1], scope='conv2d_1a_1x1')
                branch1_2 = slim.conv2d(branch1_1, 64, [5, 5], scope='conv2d_1b_5x5')
            with tf.variable_scope('Branch_2'):
                branch2_1 = slim.conv2d(net, 64, [1, 1], scope='conv2d_2a_1x1')
                branch2_2 = slim.conv2d(branch2_1, 96, [3, 3], scope='conv2d_2b_3x3')
                branch2_3 = slim.conv2d(branch2_2, 96, [3, 3], scope='conv2d_2c_3x3')
            with tf.variable_scope('Branch_3'):
                branch3_1 = slim.avg_pool2d(net, [3, 3], scope='avgPool_3a_3x3')
                branch3_2 = slim.conv2d(branch3_1, 32, [1, 1], scope='conv2d_3b_1x1')
            # 将4个分支的输出合并到一起,在第三个维度合并，即输出通道上合并
            net = tf.concat([branch_0, branch1_2, branch2_3, branch3_2], 3)

        # inception_1_m2:第一组的2号module
        with tf.variable_scope('inception_1_m2'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 64, [1, 1], scope='conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch1_1 = slim.conv2d(net, 48, [1, 1], scope='conv2d_1a_1x1')
                branch1_2 = slim.conv2d(branch1_1, 64, [5, 5], scope='conv2d_1b_5x5')
            with tf.variable_scope('Branch_2'):
                branch2_1 = slim.conv2d(net, 64, [1, 1], scope='conv2d_2a_1x1')
                branch2_2 = slim.conv2d(branch2_1, 96, [3, 3], scope='conv2d_2b_3x3')
                branch2_3 = slim.conv2d(branch2_2, 96, [3, 3], scope='conv2d_2c_3x3')
            with tf.variable_scope('Branch_3'):
                branch3_1 = slim.avg_pool2d(net, [3, 3], scope='avgPool_3a_3x3')
                branch3_2 = slim.conv2d(branch3_1, 32, [1, 1], scope='conv2d_3b_1x1')

            net = tf.concat([branch_0, branch1_2, branch2_3, branch3_2], 3)

        # inception_1_m3:第一组的3号module
        with tf.variable_scope('inception_1_m3'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 64, [1, 1], scope='conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch1_1 = slim.conv2d(net, 48, [1, 1], scope='conv2d_1a_1x1')
                branch1_2 = slim.conv2d(branch1_1, 64, [5, 5], scope='conv2d_1b_5x5')
            with tf.variable_scope('Branch_2'):
                branch2_1 = slim.conv2d(net, 64, [1, 1], scope='conv2d_2a_1x1')
                branch2_2 = slim.conv2d(branch2_1, 96, [3, 3], scope='conv2d_2b_3x3')
                branch2_3 = slim.conv2d(branch2_2, 96, [3, 3], scope='conv2d_2c_3x3')
            with tf.variable_scope('Branch_3'):
                branch3_1 = slim.avg_pool2d(net, [3, 3], scope='avgPool_3a_3x3')
                branch3_2 = slim.conv2d(branch3_1, 32, [1, 1], scope='conv2d_3b_1x1')

            net = tf.concat([branch_0, branch1_2, branch2_3, branch3_2], 3)

        # inception_2:第二个模块组，共5个inception_module
        # inception_2_m1：第二组的1号module
        with tf.variable_scope('inception_2_m1'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 384, [3, 3], stride=2,
                                       padding='VALID', scope='conv2d_0a_3x3')
            with tf.variable_scope('Branch_1'):
                branch1_1 = slim.conv2d(net, 64, [1, 1], scope='conv2d_1a_1x1')
                branch1_2 = slim.conv2d(branch1_1, 96, [3, 3], scope='conv2d_1b_3x3')
                branch1_3 = slim.conv2d(branch1_2, 96, [3, 3], stride=2,
                                        padding='VALID', scope='conv2d_1c_3x3')
            with tf.variable_scope('Branch_2'):
                branch2_1 = slim.max_pool2d(net, [3, 3], stride=2,
                                            padding='VALID', scope='maxPool_2a_3x3')

            net = tf.concat([branch_0, branch1_3, branch2_1], 3)

        # inception_2_m2：第二组的2号module
        with tf.variable_scope('inception_2_m2'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [1, 1], scope='conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch1_1 = slim.conv2d(net, 128, [1, 1], scope='conv2d_1a_1x1')
                branch1_2 = slim.conv2d(branch1_1, 128, [1, 7], scope='conv2d_1b_1x7')
                branch1_3 = slim.conv2d(branch1_2, 128, [7, 1], scope='conv2d_1c_7x1')
            with tf.variable_scope('Branch_2'):
                branch2_1 = slim.conv2d(net, 128, [1, 1], scope='conv2d_2a_1x1')
                branch2_2 = slim.conv2d(branch2_1, 128, [1, 7], scope='conv2d_2b_1x7')
                branch2_3 = slim.conv2d(branch2_2, 128, [7, 1], scope='conv2d_2c_7x1')
                branch2_4 = slim.conv2d(branch2_3, 128, [1, 7], scope='conv2d_2d_1x7')
                branch2_5 = slim.conv2d(branch2_4, 128, [7, 1], scope='conv2d_2e_7x1')
            with tf.variable_scope('Branch_3'):
                branch3_1 = slim.avg_pool2d(net, [3, 3], scope='avgPool_3a_3x3')
                branch3_2 = slim.conv2d(branch3_1, 192, [1, 1], scope='conv2d_3b_1x1')

            net = tf.concat([branch_0, branch1_3, branch2_5, branch3_2], 3)

        # inception_2_m3：第二组的3号module
        with tf.variable_scope('inception_2_m3'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [1, 1], scope='conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch1_1 = slim.conv2d(net, 160, [1, 1], scope='conv2d_1a_1x1')
                branch1_2 = slim.conv2d(branch1_1, 160, [1, 7], scope='conv2d_1b_1x7')
                branch1_3 = slim.conv2d(branch1_2, 192, [7, 1], scope='conv2d_1c_7x1')
            with tf.variable_scope('Branch_2'):
                branch2_1 = slim.conv2d(net, 160, [1, 1], scope='conv2d_2a_1x1')
                branch2_2 = slim.conv2d(branch2_1, 160, [1, 7], scope='conv2d_2b_1x7')
                branch2_3 = slim.conv2d(branch2_2, 160, [7, 1], scope='conv2d_2c_7x1')
                branch2_4 = slim.conv2d(branch2_3, 160, [1, 7], scope='conv2d_2d_1x7')
                branch2_5 = slim.conv2d(branch2_4, 192, [7, 1], scope='conv2d_2e_7x1')
            with tf.variable_scope('Branch_3'):
                branch3_1 = slim.avg_pool2d(net, [3, 3], scope='avgPool_3a_3x3')
                branch3_2 = slim.conv2d(branch3_1, 192, [1, 1], scope='conv2d_3b_1x1')

            net = tf.concat([branch_0, branch1_3, branch2_5, branch3_2], 3)

        # inception_2_m4：第二组的4号module
        with tf.variable_scope('inception_2_m4'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [1, 1], scope='conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch1_1 = slim.conv2d(net, 160, [1, 1], scope='conv2d_1a_1x1')
                branch1_2 = slim.conv2d(branch1_1, 160, [1, 7], scope='conv2d_1b_1x7')
                branch1_3 = slim.conv2d(branch1_2, 192, [7, 1], scope='conv2d_1c_7x1')
            with tf.variable_scope('Branch_2'):
                branch2_1 = slim.conv2d(net, 160, [1, 1], scope='conv2d_2a_1x1')
                branch2_2 = slim.conv2d(branch2_1, 160, [1, 7], scope='conv2d_2b_1x7')
                branch2_3 = slim.conv2d(branch2_2, 160, [7, 1], scope='conv2d_2c_7x1')
                branch2_4 = slim.conv2d(branch2_3, 160, [1, 7], scope='conv2d_2d_1x7')
                branch2_5 = slim.conv2d(branch2_4, 192, [7, 1], scope='conv2d_2e_7x1')
            with tf.variable_scope('Branch_3'):
                branch3_1 = slim.avg_pool2d(net, [3, 3], scope='avgPool_3a_3x3')
                branch3_2 = slim.conv2d(branch3_1, 192, [1, 1], scope='conv2d_3b_1x1')

            net = tf.concat([branch_0, branch1_3, branch2_5, branch3_2], 3)

        # inception_2_m4：第二组的5号module
        with tf.variable_scope('inception_2_m5'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [1, 1], scope='conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch1_1 = slim.conv2d(net, 160, [1, 1], scope='conv2d_1a_1x1')
                branch1_2 = slim.conv2d(branch1_1, 160, [1, 7], scope='conv2d_1b_1x7')
                branch1_3 = slim.conv2d(branch1_2, 192, [7, 1], scope='conv2d_1c_7x1')
            with tf.variable_scope('Branch_2'):
                branch2_1 = slim.conv2d(net, 160, [1, 1], scope='conv2d_2a_1x1')
                branch2_2 = slim.conv2d(branch2_1, 160, [1, 7], scope='conv2d_2b_1x7')
                branch2_3 = slim.conv2d(branch2_2, 160, [7, 1], scope='conv2d_2c_7x1')
                branch2_4 = slim.conv2d(branch2_3, 160, [1, 7], scope='conv2d_2d_1x7')
                branch2_5 = slim.conv2d(branch2_4, 192, [7, 1], scope='conv2d_2e_7x1')
            with tf.variable_scope('Branch_3'):
                branch3_1 = slim.avg_pool2d(net, [3, 3], scope='avgPool_3a_3x3')
                branch3_2 = slim.conv2d(branch3_1, 192, [1, 1], scope='conv2d_3b_1x1')

            net = tf.concat([branch_0, branch1_3, branch2_5, branch3_2], 3)
        # 将inception_2_m5存储到end_points中，作为Auxiliary Classifier辅助模型的分类
        end_points['inception_2_m5'] = net

        # inception_3:第三个模块组，共3个inception_module
        # inception_3_m1：第三组的1号module
        with tf.variable_scope('inception_3_m1'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 192, [1, 1], scope='conv2d_0a_1x1')
                branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2,
                                       padding='VALID', scope='conv2d_0b_3x3')
            with tf.variable_scope('Branch_1'):
                branch1_1 = slim.conv2d(net, 192, [1, 1], scope='conv2d_1a_1x1')
                branch1_2 = slim.conv2d(branch1_1, 192, [1, 7], scope='conv2d_1b_1x7')
                branch1_3 = slim.conv2d(branch1_2, 192, [7, 1], scope='conv2d_1c_7x1')
                branch1_4 = slim.conv2d(branch1_3, 192, [3, 3], stride=2,
                                        padding='VALID', scope='conv2d_1c_3x3')
            with tf.variable_scope('Branch_2'):
                branch2_1 = slim.max_pool2d(net, [3, 3], stride=2,
                                            padding='VALID', scope='maxPool_3a_3x3')

            net = tf.concat([branch_0, branch1_4, branch2_1], 3)

        # inception_3_m2：第三组的2号module
        with tf.variable_scope('inception_3_m2'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 320, [1, 1], scope='conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch1_1 = slim.conv2d(net, 384, [1, 1], scope='conv2d_1a_1x1')
                branch1_2 = tf.concat([
                    slim.conv2d(branch1_1, 384, [1, 3], scope='conv2d_1b_1x3'),
                    slim.conv2d(branch1_1, 384, [3, 1], scope='conv2d_1b_3x1')
                ], 3)
            with tf.variable_scope('Branch_2'):
                branch2_1 = slim.conv2d(net, 488, [1, 1], scope='conv2d_2a_1x1')
                branch2_2 = slim.conv2d(branch2_1, 384, [3, 3], scope='conv2d_2b_3x3')
                branch2_3 = tf.concat([
                    slim.conv2d(branch2_2, 384, [1, 3], scope='conv2d_2c_1x3'),
                    slim.conv2d(branch2_2, 384, [3, 1], scope='conv2d_2c_3x1')
                ], 3)
            with tf.variable_scope('Branch_3'):
                branch3_1 = slim.avg_pool2d(net, [3, 3], scope='avgPool_3a_3x3')
                branch3_2 = slim.conv2d(branch3_1, 192, [1, 1], scope='conv2d_3b_1x1')

            net = tf.concat([branch_0, branch1_2, branch2_3, branch3_2], 3)

        # inception_3_m3：第三组的3号module
        with tf.variable_scope('inception_3_m3'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(net, 320, [1, 1], scope='conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
                branch1_1 = slim.conv2d(net, 384, [1, 1], scope='conv2d_1a_1x1')
                branch1_2 = tf.concat([
                    slim.conv2d(branch1_1, 384, [1, 3], scope='conv2d_1b_1x3'),
                    slim.conv2d(branch1_1, 384, [3, 1], scope='conv2d_1b_3x1')
                ], 3)
            with tf.variable_scope('Branch_2'):
                branch2_1 = slim.conv2d(net, 488, [1, 1], scope='conv2d_2a_1x1')
                branch2_2 = slim.conv2d(branch2_1, 384, [3, 3], scope='conv2d_2b_3x3')
                branch2_3 = tf.concat([
                    slim.conv2d(branch2_2, 384, [1, 3], scope='conv2d_2c_1x3'),
                    slim.conv2d(branch2_2, 384, [3, 1], scope='conv2d_2c_3x1')
                ], 3)
            with tf.variable_scope('Branch_3'):
                branch3_1 = slim.avg_pool2d(net, [3, 3], scope='avgPool_3a_3x3')
                branch3_2 = slim.conv2d(branch3_1, 192, [1, 1], scope='conv2d_3b_1x1')

            net = tf.concat([branch_0, branch1_2, branch2_3, branch3_2], 3)

        return net, end_points


# 第三部分：全局平均池化、softmax、Auxiliary Logits
def inception_v3(input, num_classes=1000, is_training=True, dropout_keep_prob=0.8,
                 prediction_fn=slim.softmax, spatial_squeeze=True, reuse=None, scope='inceptionV3'):
    with tf.variable_scope(scope, 'inceptionV3', [input, num_classes], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net, end_points = inception_v3_base(input, scope=scope)

            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                aux_logits = end_points['inception_2_m5']
                with tf.variable_scope('Auxiliary_Logits'):
                    aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3,
                                                 padding='VALID', scope='AvgPool_1a_5x5')
                    aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='conv2d_1b-1x1')
                    aux_logits = slim.conv2d(aux_logits, 768, [5, 5],
                                             weights_initializer=trunc_normal(0.001),
                                             padding='VALID', scope='conv2d_2a_5x5')
                    aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1],
                                             activation_fn=None, normalizer_fn=None,
                                             weights_initializer=trunc_normal(0.001),
                                             padding='VALID', scope='conv2d_2b_1x1')
                    if spatial_squeeze:
                        aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                    end_points['Auxiliary_Logits'] = aux_logits

            with tf.variable_scope('Logits'):
                net = slim.avg_pool2d(net, [8, 8], padding='VALID', scope='avgPool_1a_8x8')
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='dropout_1b')
                end_points['PreLogits'] = net
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                '''
                类似reshape，生成一维的张量输出
                squeeze(input, axis=None, name=None, squeeze_dims=None)
                给定张量输入，此操作返回相同类型的张量，并删除所有维度为1的维度。 如果不想删除所有维度1维度，
                可以通过指定squeeze_dims来删除特定维度1维度。
                input：A Tensor。输入要挤压。
                axis：一个可选列表ints。默认为[]。如果指定，只能挤压列出的尺寸。维度索引从0开始。压缩非1的维度是错误的。必须在范围内[-- rank(input), rank(input))。
                name：操作的名称(可选)。
                squeeze_dims：现在是轴的已弃用的关键字参数。
                函数返回值：
                Tensor。与输入类型相同。 包含与输入相同的数据，但具有一个或多个删除尺寸1的维度。
                '''
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points


def time_compute(session, target, info_string):
    num_batch = 100
    num_step_burn_in = 10  # 预热轮数，头几轮迭代有显存加载、cache命中等问题可以因此跳过
    total_duration = 0.0
    total_duration_squard = 0.0
    for i in range(num_batch + num_step_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_step_burn_in:
            if i % 10 == 0:
                print('%s, step %d, duration=%.5f' % (datetime.datetime.now(), i-num_step_burn_in, duration))
            total_duration += duration
            total_duration_squard += duration * duration
    time_mean = total_duration / num_batch
    time_variance = total_duration_squard / num_batch - time_mean * time_mean
    time_stddev = math.sqrt(time_variance)
    print("%s: %s across %d steps,%.3f +/- %.3f sec per batch " %
          (datetime.datetime.now(), info_string, num_batch, time_mean, time_stddev))


def main():
    with tf.Graph().as_default():
        bath_size = 32
        height, weight = 299, 299
        input = tf.random.uniform((bath_size, height, weight, 3))
        with slim.arg_scope(inception_v3_arg_scope()):
            logits, end_points = inception_v3(input, is_training=False)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        # writer = tf.summary.FileWriter('logs/', sess.graph)
        sess.run(init)
        time_compute(sess, logits, 'Forward')


if __name__=='__main__':
    main()



