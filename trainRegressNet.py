##############################
## train a regression network that maps the models learnt from small samples to models learnt from large samples
import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
import pdb
from sklearn.preprocessing import normalize
import sys
import pdb
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from Dataset import Dataset
from function_module import regress_net, get_variable, load, snapshot_npy
rng = numpy.random
# Import arguments
parser = argparse.ArgumentParser()
logs_path = './logs'
output_dir = './models'
parser.add_argument('--pair', type=str, required=True)
parser.add_argument('--regmodel', type = str, default= "", \
                     help='pretrained model')
args = parser.parse_args()


# Parameter
learning_rate = 0.01
num_steps = 50000
batch_size = 1000
display_step = 200
# Network Parameters
num_input = 577 #  data input (64 * 3 * 3 + 1)
n_hidden_1 = 1024 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 1024 # 3rd layder number of neurous
alpha = 0.01 # alpha for leaky relu

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_input])
phase = tf.placeholder(tf.bool, name='phase')
global_step = tf.Variable(0, trainable=False)

regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)


# define loss and optimizer
y_pred = regress_net(X,phase, num_input, n_hidden_1, n_hidden_2, n_hidden_3)
y_true = Y
loss_op = tf.reduce_mean(tf.pow(y_true - y_pred,  2))
#reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

total_loss =  loss_op# + reg_loss
lr = tf.train.exponential_decay(learning_rate, global_step,
                                1000, 0.95, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(total_loss, global_step=global_step)

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss_op)
tf.summary.scalar("origin weight", tf.reduce_mean(X))
tf.summary.scalar("target weight", tf.reduce_mean(Y))
tf.summary.scalar("output weight", tf.reduce_mean(y_pred))
tf.summary.histogram('histogram', get_variable('fc1', 'dense', 'weights'))
tf.summary.histogram('histogram', get_variable('fc2', 'dense', 'weights'))
tf.summary.histogram('histogram', get_variable('fc3', 'dense', 'weights'))
tf.summary.histogram('histogram', get_variable('fc4', 'dense', 'weights'))
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    if (args.regmodel != ""):
        load(args.model, sess, ignore_missing=True)
     # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
    data = Dataset(args.pair, val = True)
    for step in range(1, num_steps+1):
        batch_x, batch_y = data.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, phase: True})

        if step % display_step == 0 or step == 1:
            [tl, lr_r, summary] = sess.run([ total_loss, lr, merged_summary_op], \
                                                 feed_dict={X: batch_x, Y: batch_y, phase: True})

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}, lr= {:.5f}".format(tl, lr_r))
            batch_x, batch_y = data.val_set()
            [loss] = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y, phase: False})
            [loss2] = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y, phase: True})
            print("Testing loss: {:.6f} :{:.6f}".format(loss, loss2));

        if (step % 10000 == 0):
            snapshot_npy(sess, output_dir, "final_", step)
    print("Optimization Finished!")
    # Testing
    batch_x, batch_y = data.val_set()
    [loss] = sess.run([total_loss], feed_dict={X: batch_x, Y: batch_y, phase: False})
    print("Testing loss: {:.6f}".format(loss));

