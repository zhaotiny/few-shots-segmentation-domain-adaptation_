import numpy as np
import os.path
import argparse
import re
logs_path = './logs_ds'
output_dir = './models'
import glob
import sys
import pdb
import tensorflow as tf
import numpy
from function_module import visualize, prediction_net, cal_loss, cal_dist_loss, \
    sparse_conv, get_variable, regularization_loss, get_train_var, snapshot_npy, \
    snapshot_npy2
from Dataset_f2gt import Dataset_f2gt
from Dataset_f2ds import Dataset_f2ds

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--f2gt', type = str, required = True,
                    help = 'txt file that has features and GT pairs')
parser.add_argument('--f2ds', type = str, default= '../cityscape/city_f2ds.txt', required = False,
                    help = 'txt file that has features and overall distribution pairs')
parser.add_argument('--conv', type = str, required = True,
                    help = 'the target weights from regression network')
parser.add_argument('--normal', action='store_false', required = False,
                    help = 'use regressed weights as regularization if specified')
parser.add_argument('--ld', type = str, default = None, required = False,
                    help = 'lambda to control weights between regularization and segmetation loss')
parser.add_argument('--name', type = str, default = "ting.npy",
                    help = 'full path name to save the model')
args = parser.parse_args()

# create or delete other files in tensorboard dir
if (not os.path.exists(logs_path)):
    os.mkdir(logs_path)
files = glob.glob(logs_path + '/*')
for file in files:
    if (os.path.isfile(file)):
        os.remove(file)

# Parameter
height = 480
width = 640
n_channel = 64 #feature channels
learning_rate = 0.001
num_steps = 2500
NUM_CLASSES = 6
display_step = 300
num_input = 577 #  data input (64 * 3 * 3 + 1)

if (args.ld != None):
    lambda1 =  float(args.ld)
else:
    lambda1 = 1#0.005
lambda2 = 0.5 # weights to balance segmentation loss and overall distribution loss

# tf Graph input
F = tf.placeholder("float", [None, n_channel, height, width])
F2 = tf.placeholder("float", [None, n_channel, height, width])
GT = tf.placeholder("float", [None, 1, height, width])
GT2 = tf.placeholder("float", [None, NUM_CLASSES])
W2 = tf.placeholder("float", [3, 3, 64, NUM_CLASSES])
B2 = tf.placeholder("float", [NUM_CLASSES])
phase2 = tf.placeholder(tf.bool, name='phase2')
global_step = tf.Variable(0, trainable=False)
# decreasing learning rate
lr = tf.train.exponential_decay(learning_rate, global_step,
                                1000, 0.85, staircase=True)
## input data
data_f2gt = Dataset_f2gt(args.f2gt);
data_f2ds = Dataset_f2ds(args.f2ds);

# network architecture
W2_tmp = tf.transpose(W2, (2, 0, 1, 3))
W2_tmp = tf.reshape(W2_tmp, (-1, NUM_CLASSES))
W2_tmp = tf.transpose(W2_tmp, (1, 0))
B2_tmp = tf.expand_dims(B2, 1)
w_pred_denorm = tf.concat([W2_tmp, B2_tmp],  1)

#prediction directly using regressed weights
y_pred_seg_pred = prediction_net(F, w_pred_denorm)
y_pred_seg_pred_prob = tf.nn.softmax(y_pred_seg_pred, dim = 1)
#final prediction
y_pred_seg_pred_fine = sparse_conv(F, NUM_CLASSES, 3, phase2, "conv")

# calculate the overall layout distribution using prediction output
y_pred_seg_pred2 = sparse_conv(F2, NUM_CLASSES, 3, phase2, "conv", reuse = True)
y_pred_seg_pred_prob2 = tf.nn.softmax(y_pred_seg_pred2, dim = 1)
max_pred_prob2 = tf.reduce_max(y_pred_seg_pred_prob2, axis = 1, keep_dims = True)
bin_pred_prob2 = y_pred_seg_pred_prob2 / max_pred_prob2
bin_pred_prob2 = bin_pred_prob2 ** 6
y_pred_seg_pred_dist = tf.reduce_mean(bin_pred_prob2, axis = [2, 3])
y_pred_seg_pred_dist = y_pred_seg_pred_dist / tf.reduce_sum(y_pred_seg_pred_dist, axis = 1, keep_dims= True)


#define loss and optimizer
#overall distribution loss
loss_dist_op = cal_dist_loss(y_pred_seg_pred_dist, GT2)
# segmentaion loss
loss_seg_op = cal_loss(y_pred_seg_pred_fine, GT, NUM_CLASSES, playing = False)
#regularization loss
conv_W = get_variable("conv", "sparse", "weights")
loss_reg_op = regularization_loss(w_pred_denorm, conv_W, lambda1, normal = args.normal)
##total loss
total_loss = loss_reg_op + loss_seg_op + lambda2 * loss_dist_op

conv_variables = get_train_var("conv")
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(total_loss, global_step=global_step, var_list=conv_variables)


##Create an visualization for showing the segmentation output of validation images

# visualization
label_vis = tf.py_func(visualize, [GT, tf.constant(NUM_CLASSES)],tf.float32)
label_vis = tf.reshape(label_vis, (-1, height, width, 3))

y_pred_seg_pred_vis = tf.py_func(visualize, [y_pred_seg_pred, tf.constant(NUM_CLASSES)],tf.float32)
y_pred_seg_pred_vis = tf.reshape(y_pred_seg_pred_vis, (-1, height, width, 3))

y_pred_seg_pred_fine_vis = tf.py_func(visualize, [y_pred_seg_pred_fine, tf.constant(NUM_CLASSES)],tf.float32)
y_pred_seg_pred_fine_vis = tf.reshape(y_pred_seg_pred_fine_vis, (-1, height, width, 3))

## logs
tf.summary.scalar("regression_loss", loss_reg_op)
tf.summary.scalar("seg loss", loss_seg_op)
tf.summary.scalar("dist loss", loss_dist_op)
tf.summary.scalar("total loss", total_loss)
tf.summary.image('b label', label_vis, max_outputs=1)
tf.summary.image('c pred', y_pred_seg_pred_vis, max_outputs=1)
tf.summary.image('d fine', y_pred_seg_pred_fine_vis, max_outputs=1)

merged_summary_op = tf.summary.merge_all()

# Start training
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
gpu_options = tf.GPUOptions()
init = tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    # Run the initializer
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
    data_conv = np.load(args.conv).item()
    convW = data_conv['conv/sparse']['weights']
    convB = data_conv['conv/sparse']['biases']


    for step in range(1, num_steps+1):
        features, label = data_f2gt.next_batch(2)
        features2, label2 = data_f2ds.next_batch(2)

        sess.run(train_op, feed_dict={W2: convW, B2: convB, F: features, GT: label, F2: features2, GT2: label2, phase2: True})

        if  step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            [regl, segl, distl, tl, lr2, a_val, summary] = sess.run([loss_reg_op, loss_seg_op, loss_dist_op, total_loss, lr, y_pred_seg_pred_dist, merged_summary_op],
                            feed_dict={F: features, GT: label, F2: features2, GT2: label2, phase2: False, W2: convW, B2: convB})
            summary_writer.add_summary(summary, step)
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.5f} + {:.5f} + {:.5f} = {:.5f}, learning rate: {:.5f}".format(regl, segl, distl,  tl, lr2))
    snapshot_npy(sess, output_dir, "final_", num_steps)
    #snapshot_npy2(sess, args.name)

#snapshot_npy(sess, num_steps)
    print("Optimization Finished!")





