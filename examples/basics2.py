# MNIST Example - network design, loss, optimizer, performance metric, train loop

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Neural Network
# Optimization variables
epochs = 10
batch_size = 100

# Training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

# Network architecture
# Weight connections
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1') # 784-in, 300-out, 300 neurons
b1 = tf.Variable(tf.random_normal([300]), name='b1') # bias for each neuron
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2') # 300-in, 10-out, 10 neurons
b2 = tf.Variable(tf.random_normal([10]), name='b2') # bias for each neuron

# Network connections
hidden1_output = tf.nn.relu(tf.add(tf.matmul(x, W1), b1)) # ReLU(x * W1 + b1)
output = tf.nn.softmax(tf.add(tf.matmul(hidden1_output, W2), b2)) # softmax(hidden1_output * W2 + b2)

# Loss - cross entropy
output_clipped = tf.clip_by_value(output, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(output_clipped) + (1 - y) * tf.log(1 - output_clipped), axis=1))
opt = tf.train.AdamOptimizer().minimize(cross_entropy)

# Initialization
init_op = tf.global_variables_initializer()

# Performance measure
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run
with tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})) as s:
    s.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            _, c = s.run([opt, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
    print(s.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))