import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# LetNet5网络一共7层：卷积——最大池化——卷积——最大池化——全连接——全连接——全连接
input_x = tf.placeholder('float', [None, 28*28])
output_y = tf.placeholder('float', [None, 10])

input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])

conv1 = tf.layers.conv2d(
    inputs=input_x_images,
    filters=6,
    kernel_size=[5, 5],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)

pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[2, 2],
    strides=2
)

conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=16,
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

flat = tf.reshape(pool2, [-1, 7*7*16])

dense1 = tf.layers.dense(
    inputs=flat,
    units=120,
    activation=tf.nn.relu
)

dense2 = tf.layers.dense(
    inputs=dense1,
    units=84,
    activation=tf.nn.relu
)

logits = tf.layers.dense(
    inputs=dense2,
    units=10,
    activation=tf.nn.softmax
)

# loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)
loss = -tf.reduce_sum(output_y * tf.log(logits))
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

accuracy_op = tf.metrics.accuracy(
    labels=tf.argmax(output_y, axis=1),
    predictions=tf.argmax(logits, axis=1)
)[1]
# 也可以写成sess=tf.InteractiveSession，二者的区别是tf.InteractiveSession()创建的是默认的会话，也就是说用户在单一会话的情境下，
# 不需要指明用哪个会话也不需要更改会话运行的情况下，就可以运行起来，这就是默认的好处。这样的话就是run和eval()函数可以不指明session。
sess = tf.Session()

init = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer()
)

sess.run(init)

for i in range(8000):
    batch = mnist.train.next_batch(100)
    sess.run(train_op, feed_dict={input_x: batch[0], output_y: batch[1]})
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy_op, feed_dict={input_x: batch[0], output_y: batch[1]})
        print('Step {}, training accuracy: {}'.format(i, train_accuracy))

print('Test accuracy: {}'.format(sess.run(accuracy_op,
                                          feed_dict={
                                              input_x: mnist.test.images,
                                              output_y: mnist.test.labels})))
# saver = tf.train.Saver()
# saver.save(sess, 'model_test')

test_output = sess.run(logits, {input_x: mnist.test.images[: 20]})
inferenced_y = np.argmax(test_output, 1)
print('Inferenced numbers: {}', inferenced_y)
print('Real numbers: ', np.argmax(mnist.test.labels[: 20], 1))

sess.close()
