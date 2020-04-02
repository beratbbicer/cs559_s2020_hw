# not working, here for example network architecture

import tensorflow as tf

def create_new_conv_layer(input, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03), name=name+'_w')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    # first and last element of padding are always 1, otherwise you'd pad along samples and channels
    tmp = tf.nn.conv2d(input, weights, [1, 1, 1, 1], padding='SAME')
    tmp += bias
    output = tf.nn.relu(tmp)

    # perform max pooling
    output = tf.nn.max_pool(output, ksize=[1, pool_shape[0], pool_shape[1], 1], strides=[1, 2, 2, 1], padding='SAME')
    return output

# input = (batch_size, xshape, yshape, channels)
def oned_conv_layer(input, num_filters, name):
    filter_shape = [input.shape[1].value, input.shape[2].value, input.shape[3].value, num_filters]
    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.03), name=name+'_w')
    b = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    return tf.nn.relu(tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='VALID') + b)

# Python optimisation variables
epochs = 10
batch_size = 64

# input image data placeholder - 784 is shape of data drawn from mnist.train.nextbatch()
x = tf.placeholder(tf.float32, [None, 784])
# dynamically reshape the input
x_reshaped = tf.reshape(x, [-1, 28, 28, 1])
# label placeholder
y = tf.placeholder(tf.float32, [None, 10])

# 2D Convs
l1_output = create_new_conv_layer(x_reshaped, 1, 32, [5, 5], [2, 2], name='l1')
l2_output = create_new_conv_layer(l1_output, 32, 64, [5, 5], [2, 2], name='l2')

# 1D Convs
l3_output = oned_conv_layer(l2_output, num_filters = 2048, name = '1dconv_1')
l4_output = oned_conv_layer(l3_output, num_filters = 512, name = '1dconv_2')
l5_output = oned_conv_layer(l4_output, num_filters = 10, name = '1dconv_3')

# Softmax output
preds = tf.nn.softmax(l5_output)

# Loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y))

# Optimizer
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
nt = {'GPU': 0})) as sess:
    # initialise the variables
    sess.run(init_op)
    total_batch = int(len(x_train.shape[0]) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = get_next_batch(batch_size)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "test accuracy: {:.3f}".format(test_acc))
    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))