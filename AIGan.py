import tensorflow as tf #machine learning
import numpy as np #matrix math
import datetime #logging the time for model checkpoints and training
import matplotlib.pyplot as plt #visualize results


#Step 1 - Collect dataset
#MNIST - handwritten character digits ~50K training and validation images + labels, 10K testing
from tensorflow.examples.tutorials.mnist import input_data
#will ensure that the correct data has been downloaded to your 
#local training folder and then unpack that data to return a dictionary of DataSet instances.
mnist = input_data.read_data_sets("MNIST_data/")

def discriminator(image, reuse=False):
    if (reuse):
        tf.get_variable_scope().reuse_variables()

    weights1 = tf.get_variable('discriminator_weights1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
    bais1 = tf.get_variable('discriminator_bais1', [32], initializer=tf.constant_initializer(0))
    layer1 = tf.nn.conv2d(input=image, filter=weights1, strides=[1, 1, 1, 1], padding='SAME')
    layer1 = layer1 + bais1
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.nn.avg_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    weights2 = tf.get_variable('discriminator_weights2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    bais2 = tf.get_variable('discriminator_bais2', [64], initializer=tf.constant_initializer(0))
    layer2 = tf.nn.conv2d(input=layer1, filter=weights2, strides=[1, 1, 1, 1], padding='SAME')
    layer2 = layer2 + bais2
    layer2 = tf.nn.relu(layer2)
    layer2 = tf.nn.avg_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    weights3 = tf.get_variable('discriminator_weights3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
    bais3 = tf.get_variable('discriminator_bais3', [1024], initializer=tf.constant_initializer(0))
    layer3 = tf.reshape(layer2, [-1, 7 * 7 * 64])
    layer3 = tf.matmul(layer3, weights3)
    layer3 = layer3 + bais3
    layer3 = tf.nn.relu(layer3)

    weights4 = tf.get_variable('discriminator_weights4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    bais4 = tf.get_variable('discriminator_bais4', [1], initializer=tf.constant_initializer(0))

    layer4 = tf.matmul(layer3, weights4) + bais4
    return layer4

def generator(batch_size, dimension):
    z = tf.truncated_normal([batch_size, dimension], mean=0, stddev=1, name='z')
    weights1 = tf.get_variable('generator_weights1', [dimension, 3136], dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
    bais1 = tf.get_variable('generator_bais1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))

    layer1 = tf.matmul(z, weights1) + bais1
    layer1 = tf.reshape(layer1, [-1, 56, 56, 1])
    layer1 = tf.contrib.layers.batch_norm(layer1, epsilon=1e-5, scope='bn1')
    layer1 = tf.nn.relu(layer1)

    weights2 = tf.get_variable('generator_weights2', [3, 3, 1, dimension/2], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    bais2 = tf.get_variable('generator_bais2', [dimension/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    layer2 = tf.nn.conv2d(layer1, weights2, strides=[1, 2, 2, 1], padding='SAME')
    layer2 = layer2 + bais2
    layer2 = tf.contrib.layers.batch_norm(layer2, epsilon=1e-5, scope='bn2')
    layer2 = tf.nn.relu(layer2)
    layer2 = tf.image.resize_images(layer2, [56, 56])

    weights3 = tf.get_variable('generator_weights3', [3, 3, dimension/2, dimension/4], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    bais3 = tf.get_variable('generator_bais3', [dimension/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    layer3 = tf.nn.conv2d(layer2, weights3, strides=[1, 2, 2, 1], padding='SAME')
    layer3 = layer3 + bais3
    layer3 = tf.contrib.layers.batch_norm(layer3, epsilon=1e-5, scope='bn3')
    layer3 = tf.nn.relu(layer3)
    layer3 = tf.image.resize_images(layer3, [56, 56])

    weights4 = tf.get_variable('generator_weights4', [1, 1, dimension/4, 1], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    bais4 = tf.get_variable('generator_bais4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    layer4 = tf.nn.conv2d(layer3, weights4, strides=[1, 2, 2, 1], padding='SAME')
    layer4 = layer4 + bais4
    layer4 = tf.sigmoid(layer4)

    return layer4

session = tf.Session()

batch_size = 50
dimensions = 100

x_placeholder = tf.placeholder("float", shape = [None,28,28,1], name='x_placeholder')

generator_z = generator(batch_size, dimensions) #Stores images

discriminator_x = discriminator(x_placeholder) #Stores prediction probabilities for actual images

discriminator_g = discriminator(generator_z,  reuse=True) #prediction probabilities for generated images

generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_g,
                                                                labels=tf.ones_like(discriminator_g)))

discriminator_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_x, labels=tf.fill([batch_size, 1], 0.9)))
discriminator_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_g, labels=tf.zeros_like(discriminator_g)))
discriminator_loss = discriminator_loss_real + discriminator_loss_fake

tvars = tf.trainable_variables()

discriminator_vars = [var for var in tvars if 'discriminator_' in var.name]
generator_vars = [var for var in tvars if 'generator_' in var.name]

with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
    discriminator_trainer_fake = tf.train.AdamOptimizer(0.0001).minimize(discriminator_loss_fake,
                                                                         var_list=discriminator_vars)
    discriminator_trainer_real = tf.train.AdamOptimizer(0.0001).minimize(discriminator_loss_real, var_list=discriminator_vars)
    generator_trainer = tf.train.AdamOptimizer(0.0001).minimize(generator_loss, var_list=generator_vars)

tf.summary.scalar('Generator_loss', generator_loss)
tf.summary.scalar('Discriminator_loss_real', discriminator_loss_real)
tf.summary.scalar('Discriminator_loss_fake', discriminator_loss_fake)

discriminator_real_count_ph = tf.placeholder(tf.float32)
discriminator_fake_count_ph = tf.placeholder(tf.float32)
generator_count_ph = tf.placeholder(tf.float32)

tf.summary.scalar('d_real_count', discriminator_real_count_ph)
tf.summary.scalar('d_fake_count', discriminator_fake_count_ph)
tf.summary.scalar('g_count', generator_count_ph)

discriminator_on_generated = tf.reduce_mean(discriminator(generator(batch_size, dimensions)))
discriminator_on_real = tf.reduce_mean(discriminator(x_placeholder))

tf.summary.scalar('d_on_generated_eval', discriminator_on_generated)
tf.summary.scalar('d_on_real_eval', discriminator_on_real)

images_for_tensorboard = generator(batch_size, dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 10)
merged = tf.summary.merge_all()
logdir = "tensorboard/gan/"
writer = tf.summary.FileWriter(logdir, session.graph)
print(logdir)


saver = tf.train.Saver()

session.run(tf.global_variables_initializer())

generator_Loss = 0
discriminatorLossFake, discriminatorLossReal = 1, 1
discriminator_real_count, discriminator_fake_count, generator_count = 0, 0, 0

for i in range(50000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    if dLossFake > 0.6:
        _, discriminatorLossReal, discriminatorLossFake, generatorLoss = sess.run([discriminator_trainer_fake,
                                                                                   discriminator_loss_real,
                                                                                   discriminator_loss_fake,
                                                                                   generator_loss],
                                                                                   {x_placeholder: real_image_batch})
        discriminator_fake_count += 1

    if gLoss > 0.5:
        # Train the generator
        _, discriminatorLossReal, discriminatorLossFake, generatorLoss = sess.run([generator_trainer,
                                                                                   discriminator_loss_real,
                                                                                   discriminator_loss_fake,
                                                                                   generator_loss],
                                                                                   {x_placeholder: real_image_batch})
        generator_count += 1

    if dLossReal > 0.45:
        # If the discriminator classifies real images as fake,
        # train discriminator on real values
        _, discriminatorLossReal, discriminatorLossFake, generatorLoss = sess.run([discriminator_trainer_real,
                                                                                   discriminator_loss_real,
                                                                                   discriminator_loss_fake,
                                                                                   generator_loss],
                                                                                   {x_placeholder: real_image_batch})
        discriminator_real_count += 1

    if i % 10 == 0:
        real_image_batch = mnist.validation.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
        summary = sess.run(merged, {x_placeholder: real_image_batch, discriminator_real_count_ph: discriminator_real_count,
                                                                                        discriminator_fake_count_ph:discriminator_fake_count,
                                                                                        generator_count_ph: generator_count})
        writer.add_summary(summary, i)
        discriminator_real_count, discriminator_fake_count, generator_count = 0, 0, 0

    if i % 1000 == 0:
        images = sess.run(generator(3, dimensions))
        discriminator_result = sess.run(discriminator(x_placeholder), {x_placeholder: images})
        print("TRAINING STEP", i, "AT", datetime.datetime.now())
        for j in range(3):
            print("Discriminator classification",
            discriminator_result[j])
            im = images[j, :, :, 0]
            plt.imshow(im.reshape([28, 28]), cmap='Greys')
            plt.show()

    if i % 5000 == 0:
        save_path = saver.save(sess, "models/pretrained_gan.ckpt", global_step=i)
        print("saved to %s" % save_path)

test_images = sess.run(generator(10, 100))
test_eval = sess.run(discriminator(x_placeholder), {x_placeholder: test_images})

real_images = mnist.validation.next_batch(10)[0].reshape([10, 28, 28, 1])
real_eval = sess.run(discriminator(x_placeholder), {x_placeholder: real_images})


for i in range(10):
    print(test_eval[i])
    plt.imshow(test_images[i, :, :, 0], cmap='Greys')
    plt.show()

for i in range(10):
    print(real_eval[i])
    plt.imshow(real_images[i, :, :, 0], cmap='Greys')
    plt.show()

