from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import BeatSynth
'''
Build a deep convolutional generative adversarial network (DCGAN) to generate digit images from a noise distribution with TensorFlow.

'''


processedData = BeatSynth.load_data()
print(len(processedData))

print(processedData[0].duration_seconds)

sound = processedData[0]
soundData = sound.get_array_of_samples()
lengthOfTransform = len(soundData)

splitted, durationOfSplit, numberOfSplits = BeatSynth.splitAudioOnDataLenght(sound, 10000)

data_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 10000])

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Params
num_steps = 10000
batch_size = 128
lr_generator = 0.002
lr_discriminator = 0.002

# Network Params
audio_dim = 2000 # 2000 audio frames * 1 channel
noise_dim = 100 # Noise data points

self.epsilon = 0.9  # Choose a random action about 10% of the time and decrease the probability of choose a random action as time goes on
self.gamma = 0.001
length_of_state = audio_dim
h_size_0 = 128
audioExpansion = 16
h_size = 128
self.num_hidden_rnn = 16
self.amount_of_data_in_each_state = 6
self.convolution_layer_1_length = 5
self.convolution_layer_2_length = 5
output_dim = len(actions) + 2

# ALWAYS INITIALIZE YOUR WEIGHTS AND BIASES
weights = {
    'wc1': tf.Variable(tf.random_normal([1, 1000, 1, h_size_0])),
    'wc2': tf.Variable(tf.random_normal([audioExpansion, 1000, h_size_0/audioExpansion, h_size])),
    'wtofc': tf.Variable(tf.random_normal([int(audioExpansion/ 2 * length_of_state / 2 * h_size), 1024])),
    #'birnn_out': tf.Variable(tf.random_normal([2 * num_hidden_rnn, output_dim]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([h_size_0])),
    'bc2': tf.Variable(tf.random_normal([h_size])),
    'out': tf.Variable(tf.random_normal([output_dim]))
}

# Build Networks
# Network Inputs
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_audio_input = tf.placeholder(tf.float32, shape=[None, 2000])
# A boolean to indicate batch normalization if it is training or inference time
is_training = tf.placeholder(tf.bool)


# LeakyReLU activation
def leakyrelu(x, alpha=0.2):
    return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)

def conv2d(x, W, b, stride=1):
    x = tf.nn.conv2d(input=x, filter=W, strides=[1, stride, stride, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return leakyrelu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

# Discriminator Network
# Input: Audio, Output: Prediction Real/Fake Beat
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tf.reshape(x, shape=[-1, 1, length_of_state, 1])

        # Convolution Layer
        x = conv2d(x, weights['wc1'], biases['bc1'])
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.reshape(x, shape=[-1, audioExpansion, length_of_state, h_size_0/audioExpansion])
        # Another Convolution Layer
        x = conv2d(x, weights['wc2'], biases["bc2"])
        # Max Pooling (down-sampling)       #Cuts both dimensiosn in half
        x = maxpool2d(x, k=2)
        x = tf.reshape(x, shape=[-1, int(audioExpansion/ 2 * length_of_state / 2 * h_size)])
        x = tf.layers.dense(x, 1024)
        x = tf.layers.batch_normalization(x, training=is_training)
        # Output 2 classes: Real and Fake Audio
        x = tf.layers.dense(x, 2)
    return x

# Generator Network
# Input: Noise, Output: Audio
# Note that batch normalization has different behavior at training and inference time,
# we then use a placeholder to indicates the layer if we are training or not.
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        x = tf.layers.dense(x, units=2000)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        # Reshape to a 4-D array of images: (batch, audio)
        # New shape: (batch, 7, 7, 128)
        x = tf.reshape(x, shape=[-1, 2000, 1])
        # Deconvolution, image shape: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 5, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        # Deconvolution, image shape: (batch, 28, 28, 1)
        x = tf.layers.conv2d_transpose(x, 1, 5, strides=2, padding='same')
        # Apply tanh for better stability - clip values to [-1, 1].
        x = tf.nn.tanh(x)
        return x


# Build Generator Network
gen_sample = generator(noise_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)

# Build the stacked generator/discriminator
stacked_gan = discriminator(gen_sample, reuse=True)

# Build Loss (Labels for real images: 1, for fake images: 0)
# Discriminator Loss for real and fake samples
disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_real, labels=tf.ones([batch_size], dtype=tf.int32)))
disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_fake, labels=tf.zeros([batch_size], dtype=tf.int32)))
# Sum both loss
disc_loss = disc_loss_real + disc_loss_fake
# Generator Loss (The generator tries to fool the discriminator, thus labels are 1)
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=stacked_gan, labels=tf.ones([batch_size], dtype=tf.int32)))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_generator, beta1=0.5, beta2=0.999)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr_discriminator, beta1=0.5, beta2=0.999)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# Discriminator Network Variables
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

# Create training operations
# TensorFlow UPDATE_OPS collection holds all batch norm operation to update the moving mean/stddev
gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')
# `control_dependencies` ensure that the `gen_update_ops` will be run before the `minimize` op (backprop)
with tf.control_dependencies(gen_update_ops):
    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')
with tf.control_dependencies(disc_update_ops):
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
sess = tf.Session()

# Run the initializer
sess.run(init)

# Training
for i in range(1, num_steps + 1):

    # Prepare Input Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    batch_x, _ = mnist.train.next_batch(batch_size)
    batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
    # Rescale to [-1, 1], the input range of the discriminator
    batch_x = batch_x * 2. - 1.

    # Discriminator Training
    # Generate noise to feed to the generator
    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
    _, dl = sess.run([train_disc, disc_loss], feed_dict={real_image_input: batch_x, noise_input: z, is_training: True})

    # Generator Training
    # Generate noise to feed to the generator
    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
    _, gl = sess.run([train_gen, gen_loss], feed_dict={noise_input: z, is_training: True})

    if i % 500 == 0 or i == 1:
        print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))



# Testing
# Generate images from noise, using the generator network.
n = 6
canvas = np.empty((28 * n, 28 * n))
for i in range(n):
    # Noise input.
    z = np.random.uniform(-1., 1., size=[n, noise_dim])
    # Generate image from noise.
    g = sess.run(gen_sample, feed_dict={noise_input: z, is_training:False})
    # Rescale values to the original [0, 1] (from tanh -> [-1, 1])
    g = (g + 1.) / 2.
    # Reverse colours for better display
    g = -1 * (g - 1)
    for j in range(n):
        # Draw the generated digits
        canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

plt.figure(figsize=(n, n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()