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
import caffe
import pdb
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from Dataset import Dataset
rng = numpy.random
# Import arguments
parser = argparse.ArgumentParser()
logs_path = './tensorflow_logs/example/'
parser.add_argument('--pair', type=str, required=True)
args = parser.parse_args()

# Parameter
learning_rate = 0.01
num_steps = 50000
batch_size = 1000
display_step = 20
# Network Parameters
num_input = 577 #  data input (64 * 3 * 3 + 1)
n_hidden_1 = 1024 # 1st layer number of neurons
n_hidden_2 = 4096 # 2nd layer number of neurons
n_hidden_3 = 1024 # 3rd layder number of neurous
num_input = 577 # output total classes (0-9 digits)
dropout = 0.5
alpha = 0 #0.01

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_input])
global_step = tf.Variable(0, trainable=False)

def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'hout': tf.Variable(tf.random_normal([n_hidden_3, num_input]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'bout': tf.Variable(tf.random_normal([num_input]))
}
'''
init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
init_biases = tf.constant_initializer(0.0)
weights = {
    'h1': tf.get_variable('h1', [num_input, n_hidden_1], initializer=init_weights),
    'h2': tf.get_variable('h2', [n_hidden_1, n_hidden_2], initializer=init_weights),
    'h3': tf.get_variable('h3', [n_hidden_2, n_hidden_3], initializer=init_weights),
    'hout': tf.get_variable('hout', [n_hidden_3, num_input], initializer=init_weights)
}
biases = {
    'b1': tf.get_variable('b1', [n_hidden_1], initializer=init_biases),
    'b2': tf.get_variable('b2', [n_hidden_2], initializer=init_biases),
    'b3': tf.get_variable('b3', [n_hidden_3], initializer=init_biases),
    'bout': tf.get_variable('bout', [num_input], initializer=init_biases)
}
def neural_net(x):
    fc1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    fc1 = lrelu(fc1, alpha)
    #fc1 = tf.nn.dropout(fc1, dropout)
    fc2 = tf.add(tf.matmul(fc1, weights['h2']), biases['b2'])
    fc2 = lrelu(fc2, alpha)
    #fc2 = tf.nn.dropout(fc2, dropout)
    fc3 = tf.add(tf.matmul(fc2, weights['h3']), biases['b3'])
    fc3 = lrelu(fc3, alpha)
   # fc3 = tf.nn.dropout(fc3, dropout)
    output_layer = tf.matmul(fc3, weights['hout'])+ biases['bout']
    output_layer = tf.nn.sigmoid(output_layer)
    return output_layer
'''

def neural_net(x):
    fc1 = tf.nn.xw_plus_b(x, weights['h1'], biases['b1'])
    fc1 = tf.nn.sigmoid(fc1)

    fc2 = tf.nn.xw_plus_b(fc1, weights['h2'], biases['b2'])
    fc2 = tf.nn.sigmoid(fc2)

    fc3 = tf.nn.xw_plus_b(fc2, weights['h3'], biases['b3'])
    fc3 = tf.nn.sigmoid(fc3)

    fc4 = tf.nn.xw_plus_b(fc3, weights['hout'], biases['bout'])
    fc4 = tf.nn.sigmoid(fc4)
    return fc4

# define loss and optimizeer
y_pred = neural_net(X)
y_true = Y
loss_op = tf.reduce_mean(tf.pow(y_true - y_pred,  2))
lr = tf.train.exponential_decay(learning_rate, global_step,
                                1000, 0.95, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss_op, global_step=global_step)

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss_op)
tf.summary.scalar("origin weight", tf.reduce_mean(X))
tf.summary.scalar("target weight", tf.reduce_mean(Y))
tf.summary.scalar("output weight", tf.reduce_mean(y_pred))
tf.summary.histogram('histogram', weights['h1'])
tf.summary.histogram('histogram', weights['h2'])
tf.summary.histogram('histogram', weights['h3'])
tf.summary.histogram('histogram', weights['hout'])
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

     # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter("./logs", sess.graph)
    data = Dataset(args.pair, num_input)
    pdb.set_trace()
    for step in range(1, num_steps+1):
        batch_x, batch_y = data.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        if step % display_step == 0 or step == 1:
            [loss, pred, lr_r, summary] = sess.run([loss_op, y_pred, lr, merged_summary_op], feed_dict={X: batch_x, Y: batch_y})
            summary_writer.add_summary(summary, step)

            print("Step " + str(step) + ", mean weights: pred= " + \
                  "{:.6f}".format(np.mean(pred)) + " input= {:.6f}".format(np.mean(batch_x)) + " target= {:.6f}".format(np.mean(batch_y)))
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}, lr= {:.5f}".format(loss, lr_r))

    print("Optimization Finished!")
    summary_writer.close()


