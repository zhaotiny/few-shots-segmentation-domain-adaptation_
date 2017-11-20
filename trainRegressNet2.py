import numpy as np
import matplotlib.pyplot as plt2
import os.path
import json
import scipy
import argparse
import math
import pylab
import pdb
from sklearn.preprocessing import normalize
caffe_root = '/home/selfdriving/zhaotiny/SegNet/caffe-segnet-cudnn52/' 			# Change this to the absolute directoy to SegNet Caffe
logs_path = './logs2'
output_dir = './models'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import pdb
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from Dataset2 import Dataset2
rng = numpy.random
# Import arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--pair', type=str, required=True)
parser.add_argument('--regmodel', type = str, default= "",  \
                     help='pretrained model')
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

# Parameter
height = 480
width = 640
n_channel = 64
learning_rate = 0.005
num_steps = 10000
batch_size = 6
NUM_CLASSES = batch_size
display_step = 5
# Network Parameters
num_input = 577 #  data input (64 * 3 * 3 + 1)
n_hidden_1 = 1024 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 1024 # 3rd layder number of neurous
alpha = 0.01 # alpha for leaky relu
lambda1 = 10.0
lambda_min = 1.0

# tf Graph input
X = tf.placeholder("float", [batch_size, num_input])
Y = tf.placeholder("float", [batch_size, num_input])
F = tf.placeholder("float", [None, n_channel, height, width])
GT = tf.placeholder("float", [None, 1, height, width])
W2 = tf.placeholder("float", [3, 3, 64, NUM_CLASSES])
B2 = tf.placeholder("float", [NUM_CLASSES])
IMG = tf.placeholder("float", [None, height, width, 3])
LABEL = tf.placeholder("float", [None, height, width, 1])
phase = tf.placeholder(tf.bool, name='phase')
global_step = tf.Variable(0, trainable=False)
## decreasing weights for regression loss
lambda_tf = tf.train.exponential_decay(lambda1, global_step,
                                       1000, 0.8, staircase = True)
lambda_tf = tf.maximum(tf.constant(lambda_min), lambda_tf)
# decreasing learning rate
lr = tf.train.exponential_decay(learning_rate, global_step,
                                1000, 0.95, staircase=True)
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

## define conv2d with our own weights and features
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME', data_format = "NCHW")
    x = tf.nn.bias_add(x, b, data_format = "NCHW")
    return x

def weighted_loss(logits, labels, num_classes, head=None):
    """ median-frequency re-weighting """
    #with tf.name_scope('loss'):
    logits = tf.transpose(logits, (0, 2, 3, 1))
    labels = tf.transpose(labels, (0, 2, 3, 1))

    logits = tf.reshape(logits, (-1, num_classes))

    epsilon = tf.constant(value=1e-10)

    logits = logits + epsilon

    # consturct one-hot label array
    label_flat = tf.reshape(labels, (-1, 1))

    # should be [batch ,num_classes]
    labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

    softmax = tf.nn.softmax(logits)

    cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), axis=[1])

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    return cross_entropy_mean

def cal_loss(logits, labels):
    loss_weight = np.array([
        0.2533,
        3.1525,
        3.4525,
        21.3407,
        32.5793,
        0.1050]) # class 0~11

    labels = tf.cast(labels, tf.int32)
    # return loss(logits, labels)
    return weighted_loss(logits, labels, num_classes=NUM_CLASSES, head=loss_weight)

## regression network that maps small smaples model to large sample models
def regress_net(x):
    fc1 = dense_batch_relu(x, n_hidden_1, phase, "fc1")
    fc2 = dense_batch_relu(fc1, n_hidden_2, phase, "fc2")
    fc3 = dense_batch_relu(fc2, n_hidden_3, phase, "fc3")
    fc4 = dense(fc3, num_input, "fc4")
    fc4 = tf.nn.sigmoid(fc4)
    return fc4

## conv the models with the features to obtain the prediction
def prediction_net(y_pred):
    W = y_pred[:, 0: 576]
    B = y_pred[:, 576]
    W = tf.transpose(W, (1, 0))
    W = tf.reshape(W, (64, 3, 3, -1))
    W = tf.transpose(W, (1, 2, 0, 3))
    pred = conv2d(F, W, B)
    return pred

## visualize the prediction output
def visualize(predicted):
    if (predicted.shape[1] == 1):
        ind = np.squeeze(predicted[0,:,:,:])
    else:
        output = np.squeeze(predicted[0, :, :, :])
        ind = np.argmax(output, axis=0)
    r = ind.copy()
    g = ind.copy()
    b = ind.copy()

    Sky = [128,128,128]
    Road = [128,64,128]
    Car = [64,0,128]
    Pedestrian = [64,64,0]
    Bicyclist = [0,128,192]
    Unlabelled = [0,0,0]

    label_colours = np.array([Road, Car, Sky, Pedestrian, Bicyclist, Unlabelled])

    for l in range(0,6):
        r[ind==l] = label_colours[l,0]
        g[ind==l] = label_colours[l,1]
        b[ind==l] = label_colours[l,2]

    rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
    rgb[:,:,0] = r/255.0
    rgb[:,:,1] = g/255.0
    rgb[:,:,2] = b/255.0
    return rgb

## input data
data = Dataset2(args.pair, val = True, defstat = True, numC = NUM_CLASSES)

# define loss and optimizer
w_pred = regress_net(X)
w_pred_denorm = tf.py_func(data.denormalize_target, [w_pred], tf.float32)
y_pred_seg_pred = prediction_net(w_pred_denorm)
y_pred_seg_pred_prob = tf.nn.softmax(y_pred_seg_pred, dim = 1)
w_true = Y

Y_denorm = tf.py_func(data.denormalize_target, [Y], tf.float32)
y_pred_seg_large = prediction_net(Y_denorm)
y_pred_seg_large_prob = tf.nn.softmax(y_pred_seg_large, dim = 1)

X_denorm = tf.py_func(data.denormalize_input, [X], tf.float32)
y_pred_seg_small = prediction_net(X_denorm)
y_pred_seg_small_prob = tf.nn.softmax(y_pred_seg_small, dim = 1)

pred_test = conv2d(F, W2, B2)
pred_test_prob = tf.nn.softmax(pred_test, dim = 1)

#Create an output for showing the segmentation output of validation images
y_pred_seg_pred_vis = tf.argmax(y_pred_seg_pred_prob, axis=1)
y_pred_seg_pred_vis = tf.cast(y_pred_seg_pred_vis, dtype=tf.float32)
y_pred_seg_pred_vis = tf.reshape(y_pred_seg_pred_vis, (-1, 1, height, width))
y_pred_seg_pred_vis = tf.transpose(y_pred_seg_pred_vis, (0, 2, 3, 1))

y_pred_seg_large_vis = tf.argmax(y_pred_seg_large, axis=1)
y_pred_seg_large_vis = tf.cast(y_pred_seg_large_vis, dtype = tf.float32)
y_pred_seg_large_vis = tf.reshape(y_pred_seg_large_vis, (-1, 1, height, width))
y_pred_seg_large_vis = tf.transpose(y_pred_seg_large_vis, (0, 2, 3, 1))

y_pred_seg_small_vis = tf.argmax(y_pred_seg_small, axis=1)
y_pred_seg_small_vis = tf.cast(y_pred_seg_small_vis, dtype = tf.float32)
y_pred_seg_small_vis = tf.reshape(y_pred_seg_small_vis, (-1, 1, height, width))
y_pred_seg_small_vis = tf.transpose(y_pred_seg_small_vis, (0, 2, 3, 1))


loss_op = tf.reduce_mean(tf.pow(w_true - w_pred,  2)) #regression loss
loss_seg_op = cal_loss(y_pred_seg_large, GT)   ##segmentaion loss
total_loss = lambda_tf * loss_op + loss_seg_op      ##total loss

optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(total_loss, global_step=global_step)

## logs
tf.summary.scalar("regression_loss", loss_op)
tf.summary.scalar("seg loss", loss_seg_op)
tf.summary.scalar("total loss", total_loss)
tf.summary.histogram('histogram fc1', get_variable('fc1', 'dense', 'weights'))
tf.summary.histogram('histogram fc2', get_variable('fc2', 'dense', 'weights'))
tf.summary.histogram('histogram fc3', get_variable('fc3', 'dense', 'weights'))
tf.summary.histogram('histogram fc4', get_variable('fc4', 'dense', 'weights'))
tf.summary.image('a ori image', IMG, max_outputs=1)
tf.summary.image('b label', LABEL, max_outputs=1)
tf.summary.image('c pred1', y_pred_seg_pred_vis, max_outputs=1)
tf.summary.image('d small', y_pred_seg_small_vis, max_outputs=1)
tf.summary.image('e large', y_pred_seg_large_vis, max_outputs=1)

merged_summary_op = tf.summary.merge_all()
init = tf.global_variables_initializer()

# Start training
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    # Run the initializer
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
    if (args.regmodel != ""):
        load(args.regmodel, sess, ignore_missing=True)
    for step in range(1, num_steps+1):
        net.forward()
        image = net.blobs['data'].data
        features = net.blobs['conv1_2_D'].data
        label = net.blobs['label'].data
        predicted = net.blobs['prob'].data

        image2 = image[:, [2, 1, 0], :,:]
        image2 = np.transpose(image2, (0, 2, 3, 1)) / 255.0
        label2 = np.transpose(label, (0, 2, 3, 1))

        batch_x, batch_y = data.next_batch()
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, F: features, GT: label, phase: True})
        convW = net.params['conv1_1_D_f'][0].data[...]
        convB = net.params['conv1_1_D_f'][1].data[...]
        convW = np.reshape(convW, (6, -1))
        convW = np.transpose(convW, (1, 0))

        convW = np.reshape(convW, (64, 3, 3, -1))
        convW = np.transpose(convW, (1, 2, 0, 3))

        if  step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy

           # [y_norm, y_denorm] = sess.run([Y, Y_denorm], \
            #                feed_dict={X: batch_x, Y: batch_y, F: features, GT: label, phase: False, W2: convW, B2: convB})

           # pdb.set_trace()

            [pred, large, small, test_tf, summary] = sess.run([y_pred_seg_pred_prob, y_pred_seg_large_prob, y_pred_seg_small_prob, pred_test_prob, merged_summary_op], \
                            feed_dict={X: batch_x, Y: batch_y, F: features, GT: label, phase: False, W2: convW, B2: convB, IMG: image2, LABEL: label2})
            summary_writer.add_summary(summary, step)
            '''
            pred = visualize(pred)
            large = visualize(large)
            small = visualize(small)
            test_tf = visualize(test_tf)
            test_caffe = visualize(predicted)

            gt = visualize(label)
            image = np.squeeze(image[0,:,:,:])
            image = image / 255.0
            image = np.transpose(image, (1,2,0))
            image = image[:,:,(2,1,0)]
       #     pdb.set_trace()
            plt2.figure()
            plt2.imshow(image,vmin=0, vmax=1)
            plt2.title("original image")
            plt2.figure()
            plt2.imshow(gt,vmin=0, vmax=1)
            plt2.title("Ground-truth")
            plt2.figure()
            plt2.imshow(pred, vmin = 0, vmax = 1)
            plt2.title("Prediction from network")
            plt2.figure()
            plt2.imshow(large,vmin=0, vmax=1)
            plt2.title("Large Weights")
            plt2.figure()
            plt2.imshow(small,vmin=0, vmax=1)
            plt2.title("Small Weights")

            plt2.figure()
            plt2.imshow(test_tf,vmin=0, vmax=1)
            plt2.title("TF Weights")
        #    plt2.figure()
        #    plt2.imshow(test_caffe,vmin=0, vmax=1)
        #    plt2.title("caffe Weights")
            plt2.show()
            '''

            [regl, segl, tl, lr2, lambda_val] = sess.run([loss_op, loss_seg_op, total_loss, lr, lambda_tf], feed_dict={X: batch_x, Y: batch_y, F: features, GT: label, phase: False})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.5f}, {:.5f}, {:.5f}, learning rate: {:.5f}, lambda: {:.2f}".format(regl, segl, tl, lr2, lambda_val))

    print("Optimization Finished!")





