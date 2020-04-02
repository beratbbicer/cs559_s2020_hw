# Variables, placeholders, operations, initializer, session, and run
# ------------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

# first, create a TensorFlow constant
const = tf.constant(2.0, name="const")
    
# create TensorFlow variables
# b = tf.Variable(2.0, name='b') # value fixed
b = tf.placeholder(tf.float32, [None, 1], name='b')
c = tf.Variable(1.0, name='c')

# create TensorFlow operations
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

# setup the variable initialisation
init_op = tf.global_variables_initializer()

# start the session
config = tf.ConfigProto(
    device_count = {'GPU': 0}
)
with tf.Session(config=config) as s:
    # initialise the variables
    s.run(init_op)
    # compute the output of the graph
    # a_out = s.run(a)
    a_out = s.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    print("Variable a is {}".format(a_out))