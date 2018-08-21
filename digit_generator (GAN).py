import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
import tensorflow as tf

np.random.seed(10)
random_dim = 100

def process_data():
    data = pd.read_csv('mnist.csv').values
    train_data = (data.astype(np.float32) - 127.5) / 127.5
    train_data = train_data.reshape(len(train_data), 784)

    return train_data

def generator_network(input, reuse=False):
    with tf.variable_scope('generator') as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        generator = tf.layers.dense(input, units=256, kernel_initializer=tf.initializers.random_normal(stddev=0.02))
        generator = tf.nn.leaky_relu(generator, alpha=0.2)

        generator = tf.layers.dense(generator, units=512)
        generator = tf.nn.leaky_relu(generator, alpha=0.2)

        generator = tf.layers.dense(generator, units=1024)
        generator = tf.nn.leaky_relu(generator, alpha=0.2)

        generator = tf.layers.dense(generator, units=784, activation=tf.tanh)

    return generator

def discriminator_network(input, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()

        discriminator = tf.layers.dense(input, units=1024, kernel_initializer=tf.initializers.random_normal(stddev=0.02))
        discriminator = tf.nn.leaky_relu(discriminator, alpha=0.2)
        discriminator = tf.layers.dropout(discriminator, 0.3)

        discriminator = tf.layers.dense(discriminator, units=512)
        discriminator = tf.nn.leaky_relu(discriminator, alpha=0.2)
        discriminator = tf.layers.dropout(discriminator, 0.3)

        discriminator = tf.layers.dense(discriminator, units=256)
        discriminator = tf.nn.leaky_relu(discriminator, alpha=0.2)
        discriminator = tf.layers.dropout(discriminator, 0.3)

        discriminator = tf.layers.dense(discriminator, units=1)

    return discriminator

def plot_images(epoch, img, examples=100, dim=(10, 10), figsize=(10, 10)):
    generated_images = img.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epoch)

def train(epoch, batch_size=128):
    tf.reset_default_graph()
    data = process_data()
    batch_count = data.shape[0] / batch_size

    sess = tf.Session()
    gen = tf.placeholder(tf.float32, shape=[None, random_dim], name='input')
    dis = tf.placeholder(tf.float32, shape=[None, 784], name='input')

    dis_real = discriminator_network(dis)
    generator = generator_network(gen)
    dis_fake = discriminator_network(generator, reuse=True)

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake, labels=tf.ones_like(dis_fake)))
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_real, labels=(tf.ones_like(dis_real)*0.9)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_fake, labels=tf.zeros_like(dis_fake)))
    d_loss = d_loss_real + d_loss_fake

    vars = tf.trainable_variables()
    d_vars = [var for var in vars if 'discriminator' in var.name]
    g_vars = [var for var in vars if 'generator' in var.name]

    trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(d_loss, var_list=d_vars)
    trainerG = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(g_loss, var_list=g_vars)

    sess.run(tf.global_variables_initializer())
    for i in xrange(1, epoch+1):
        print 'Epoch: {}'.format(i)

        for _ in tqdm(xrange(batch_count)):
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = data[np.random.randint(0, data.shape[0], size=batch_size)]
            _, dLoss = sess.run([trainerD, d_loss], feed_dict={gen: noise, dis: image_batch})
            _, gLoss = sess.run([trainerG, g_loss], feed_dict={gen: noise})

        if i == 1 or i % 20 == 0:
            print 'd_loss: {}, g_loss: {}'.format(dLoss, gLoss)
            sample_noise = np.random.normal(0, 1, size=[random_dim, random_dim])
            img = sess.run(generator, feed_dict={gen: sample_noise})
            plot_images(i, img)

if __name__ == '__main__':
    train(400, 128)