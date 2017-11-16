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
logs_path = './logs'
output_dir = './models'
parser.add_argument('--pair', type=str, required=True)
parser.add_argument('--model', type = str, default= "", \
                     help='pretrained model')
args = parser.parse_args()


# Parameter
learning_rate = 0.01
num_steps = 5000
batch_size = 1000
display_step = 1000
# Network Parameters
num_input = 577 #  data input (64 * 3 * 3 + 1)
n_hidden_1 = 1024 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 1024 # 3rd layder number of neurous
dropout = 0.5
alpha = 0.01

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_input])
phase = tf.placeholder(tf.bool, name='phase')
global_step = tf.Variable(0, trainable=False)

## leaky relu
def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

## define fully-conneted layer
def dense(x, num_out, scope):
    with tf.variable_scope(scope):
        return tf.contrib.layers.fully_connected(x, num_out,
                                             activation_fn=None,
                                             scope='dense')
## define the layer of fully-connected + batch normalization + nonlinearity
def dense_batch_relu(x, num_out, phase, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, num_out,
                                               activation_fn=None,
                                               scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1,
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn', updates_collections=None)
        return lrelu(h2, alpha)

## save the snapshot to as .npy format
def snapshot_npy(sess, iter):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data = {}
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    valid_str = ["weights", "biases", "moving_mean", "moving_variance", "beta", "gamma"]
    for i in range(len(all_variables)):
        variable_name = all_variables[i].name
        parts = variable_name.split('/');
        if (len(parts) == 3):
            scope_name = parts[0] + '/' + parts[1]
            vari_name_temp = parts[-1].split(':')[0]
            if (vari_name_temp not in valid_str):
                continue
        else:
            continue
        vari_name = parts[-1].split(':')[0]
        if (scope_name not in data.keys()):
            data[scope_name] = {}
        data[scope_name][vari_name] = sess.run(all_variables[i])
    filename = output_dir + '/' + str(iter) + '.npy'
    np.save(filename,data);
    print 'Wrote snapshot to: {:s}'.format(filename)


## load the parameters from .npy model
def load(data_path, session, ignore_missing=False):
    data_dict = np.load(data_path).item()
    for key in data_dict:
        #   if (key == 'bbox_pred'): continue;
        with tf.variable_scope(key, reuse=True):
            for subkey in data_dict[key]:
                try:
                    var = tf.get_variable(subkey)
                    session.run(var.assign(data_dict[key][subkey]))
                    print "assign pretrain model "+subkey+ " to "+key
                except ValueError:
                    print "ignore "+key
                    if not ignore_missing:
                        raise

# get variable given scope/name/weight:0
def get_variable(scope, name, weights):
    tmp = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope + '/' + name + '/' + weights + ':0')
    return tmp[0]

## define regress net
def neural_net(x):
    fc1 = dense_batch_relu(x, n_hidden_1, phase, "fc1")
    fc2 = dense_batch_relu(fc1, n_hidden_2, phase, "fc2")
    fc3 = dense_batch_relu(fc2, n_hidden_3, phase, "fc3")
    fc4 = dense(fc3, num_input, "fc4")
    fc4 = tf.nn.sigmoid(fc4)
    return fc4

# define loss and optimizer
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
    pdb.set_trace()
    if (args.model != ""):
        load(args.model, sess, ignore_missing=True)
     # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
    data = Dataset(args.pair, num_input, True)
    for step in range(1, num_steps+1):
        batch_x, batch_y = data.next_batch(batch_size)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, phase: True})

        if step % display_step == 0 or step == 1:
            [loss, pred, lr_r, summary] = sess.run([loss_op, y_pred, lr, merged_summary_op], \
                                                   feed_dict={X: batch_x, Y: batch_y, phase: True})
            if (step > 200):
                summary_writer.add_summary(summary, step)

            print("Step " + str(step) + ", mean weights: pred= " + \
                  "{:.6f}".format(np.mean(pred)) + " input= {:.6f}".format(np.mean(batch_x)) + " target= {:.6f}".format(np.mean(batch_y)))
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}, lr= {:.5f}".format(loss, lr_r))
            batch_x, batch_y = data.val_set()
            [loss] = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y, phase: False})
            [loss2] = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y, phase: True})
            print("Testing loss: {:.6f} :{:.6f}".format(loss, loss2));

    print("Optimization Finished!")
    snapshot_npy(sess, num_steps)
    # Testing
    pdb.set_trace()
    batch_x, batch_y = data.val_set()
    [loss] = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y, phase: False})
    print("Testing loss: {:.6f}".format(loss));


'''
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
'''
def neural_net(x):
    fc1 = tf.nn.xw_plus_b(x, weights['h1'], biases['b1'])
    fc1 = tf.contrib.layers.batch_norm(fc1,
                                          center=True, scale=True,
                                          is_training=phase, scope='bn1', updates_collections=None)
    fc1 = tf.nn.relu(fc1)
    fc2 = tf.nn.xw_plus_b(fc1, weights['h2'], biases['b2'])
    fc2 = tf.contrib.layers.batch_norm(fc2,
                                          center=True, scale=True,
                                          is_training=phase, scope='bn2', updates_collections=None)
    fc2 = tf.nn.relu(fc2)

    fc3 = tf.nn.xw_plus_b(fc2, weights['h3'], biases['b3'])
    fc3 = tf.contrib.layers.batch_norm(fc3,
                                          center=True, scale=True,
                                          is_training=phase, scope='bn3', updates_collections=None)
    fc3 = tf.nn.relu(fc3)
    fc4 = tf.nn.xw_plus_b(fc3, weights['hout'], biases['bout'])
    fc4 = tf.nn.sigmoid(fc4)
    return fc4
'''
