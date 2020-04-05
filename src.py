import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import matplotlib.pyplot as plt
import numpy as np

def get_label(path):
    tmp = tf.strings.split(path, '/')
    tmp = tf.strings.split(tmp[-1], '_')[0]
    return tf.strings.to_number(tmp, out_type = tf.int32)

def get_image(img):
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.rgb_to_grayscale(img)
    return img

def process(path):
    label = get_label(path)
    img = get_image(tf.io.read_file(path))
    return img, label

def prepare_for_training(ds, batch_size, cache=True, shuffle_buffer_size=1024):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    # ds = ds.repeat()
    ds = ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def get_dataset(datapath, batch_size):
    # datapath = 'data/training/'
    files = tf.data.Dataset.list_files(datapath + '*.jpg')
    dataset = files.map(process, num_parallel_calls=AUTOTUNE)
    dataset = prepare_for_training(dataset, batch_size)
    return dataset

def conv2d(input, filters, stride_size, leaky_relu_alpha):
    out = tf.nn.conv2d(input, filters, strides=[1, stride_size, stride_size, 1], padding='SAME') 
    return tf.nn.leaky_relu(out, alpha=leaky_relu_alpha) 

def maxpool(input, pool_size, stride_size):
    return tf.nn.max_pool2d(input, ksize=[1, pool_size, pool_size, 1], padding='VALID', strides=[1, stride_size, stride_size, 1])

def dense(input, weights, leaky_relu_alpha, dropout_rate):
    x = tf.nn.leaky_relu(tf.matmul(input, weights), alpha=leaky_relu_alpha)
    return tf.nn.dropout(x, rate=dropout_rate)

def get_weight(shape, name, initializer):
    return tf.Variable(initializer(shape), name=name, trainable=True, dtype=tf.float32)

def model(x, weights, alpha, dropout_rate):
    c1 = conv2d(x, weights[0], stride_size=1, leaky_relu_alpha = alpha) 
    c1 = conv2d(c1, weights[1], stride_size=1, leaky_relu_alpha = alpha) 
    p1 = maxpool(c1, pool_size=2, stride_size=2)
    
    c2 = conv2d(p1, weights[2], stride_size=1, leaky_relu_alpha = alpha)
    c2 = conv2d(c2, weights[3], stride_size=1, leaky_relu_alpha = alpha) 
    p2 = maxpool(c2, pool_size=2, stride_size=2)
    
    c3 = conv2d(p2, weights[4], stride_size=1, leaky_relu_alpha = alpha) 
    c3 = conv2d(c3, weights[5], stride_size=1, leaky_relu_alpha = alpha) 
    p3 = maxpool(c3, pool_size=2, stride_size=2)
    
    c4 = conv2d(p3, weights[6], stride_size=1, leaky_relu_alpha = alpha)
    c4 = conv2d(c4, weights[7], stride_size=1, leaky_relu_alpha = alpha)
    p4 = maxpool(c4, pool_size=2, stride_size=2)

    c5 = conv2d(p4, weights[8], stride_size=1, leaky_relu_alpha = alpha)
    c5 = conv2d(c5, weights[9], stride_size=1, leaky_relu_alpha = alpha)
    p5 = maxpool(c5, pool_size=2, stride_size=2)

    c6 = conv2d(p5, weights[10], stride_size=1, leaky_relu_alpha = alpha)
    c6 = conv2d(c6, weights[11], stride_size=1, leaky_relu_alpha = alpha)
    p6 = maxpool(c6, pool_size=2, stride_size=2)

    flatten = tf.reshape(p6, shape=(tf.shape(p6)[0], -1)) 

    # try to use 1d convolutions rather than FC layers
    d1 = dense(flatten, weights[12], leaky_relu_alpha = alpha, dropout_rate = dropout_rate) 
    d2 = dense(d1, weights[13], leaky_relu_alpha = alpha, dropout_rate = dropout_rate)
    d3 = dense(d2, weights[14], leaky_relu_alpha = alpha, dropout_rate = dropout_rate)
    d4 = dense(d3, weights[15], leaky_relu_alpha = alpha, dropout_rate = dropout_rate)

    return tf.matmul(d4, weights[16]) # regression output

def loss(target, pred):
    return tf.keras.losses.MSE(target, pred)

def train_step(model, input, labels, weights, alpha, dropout_rate):
    with tf.GradientTape() as tape:
        current_loss = loss(labels, model(input, weights, alpha, dropout_rate))
    grads = tape.gradient(current_loss, weights)
    optimizer.apply_gradients(zip(grads, weights))
    # print('Train batch loss: {:.3f}'.format(tf.reduce_mean(current_loss).numpy()))

# -----------------------------------------------------------------------------------------------------

output_classes = 8
batch_size = 256
learning_rate = 0.002
leaky_relu_alpha = 0.2
dropout_rate = 0.5
initializer = tf.initializers.glorot_uniform()
optimizer = tf.optimizers.Adam(learning_rate)
num_epochs = 10
train_size = 3555
val_size = 893
test_size = 895
batch_count = int(train_size / batch_size) + 1

shapes = [
    [3, 3, 1, 16], # 3x3 conv, 1 input, 16 output
    [3, 3, 16, 16],
    [3, 3, 16, 32],
    [3, 3, 32, 32],
    [3, 3, 32, 64],
    [3, 3, 64, 64],
    [3, 3, 64, 128],
    [3, 3, 128, 128],
    [3, 3, 128, 256],
    [3, 3, 256, 256],
    [3, 3, 256, 512],
    [3, 3, 512, 512],
    [512, 256],
    [256, 128],
    [128, 64],
    [64, 32],
    [32, 1],
]


weights = []
for i in range(len(shapes)):
    weights.append(get_weight(shapes[i], 'weight{}'.format(i), initializer))

train_data = get_dataset('data/training/', batch_size)
val_data = get_dataset('data/validation/', batch_size)
test_data = get_dataset('data/test/', batch_size)
val_losses = []
for e in range(num_epochs):
    idx = 1
    for x,y in train_data:
        print('Train batch {} out of {}'.format(idx, batch_count))
        train_step(model, x, y, weights, leaky_relu_alpha, dropout_rate)
        idx += 1
    val_loss = 0
    for x,y in val_data:
        val_loss += tf.reduce_mean(loss(y, model(x, weights, leaky_relu_alpha, dropout_rate))).numpy()
    val_loss /= batch_count
    val_losses.append(val_loss)
    print('epoch {} - validation loss: {:.3f}'.format(e + 1, val_loss))

plt.figure()
plt.plot([e for e in range(num_epochs)], val_losses)
plt.title('Validation loss vs epochs')
plt.show()
