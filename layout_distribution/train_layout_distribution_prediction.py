"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vgg19_trainable as vgg19
import utils
import pdb
from Dataset import Dataset
import glob
import os
import numpy as np
np.set_printoptions(precision=4, suppress=True)
#img1 = utils.load_image("./test_data/tiger.jpeg")
#img1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger
logs_path = './logs'
if (not os.path.exists(logs_path)):
    os.mkdir(logs_path)
files = glob.glob(logs_path + '/*')
for file in files:
    if (os.path.isfile(file)):
        os.remove(file)

NUM_CLASS = 6
total_iter = 200000
batch_size = 20
num_iter = total_iter / batch_size
display_step = 50
learning_rate = 0.0008
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
true_out = tf.placeholder(tf.float32, [None, NUM_CLASS])
train_mode = tf.placeholder(tf.bool)
global_step = tf.Variable(0, trainable=False)
init = tf.global_variables_initializer()
epsilon = tf.constant(value=1e-10)

vgg = vgg19.Vgg19('5000.npy')
vgg.build(images, train_mode)
train_layers = ['fc8', 'fc7']
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
#cost = tf.reduce_sum(- tf.log(vgg.prob + epsilon) * true_out, 1)
#cost = tf.reduce_mean(cost)
lr = tf.train.exponential_decay(learning_rate, global_step,
                                1000, 0.90, staircase=True)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = vgg.fc8, labels= true_out))
optimizer  = tf.train.AdamOptimizer(learning_rate= lr)
train_op = optimizer.minimize(cost,  global_step=global_step, var_list = var_list)

data = Dataset('train.txt')

tf.summary.scalar('loss', cost)
merge_op = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
    for step in range(1, num_iter + 1):
        X, Y = data.next_batch(batch_size)
      #  pdb.set_trace()
        sess.run(train_op, feed_dict={images: X, true_out: Y, train_mode: True})
        if (step % display_step == 0 or step == 1):
            [loss, lr_val, prob_val, gt_val, merge] = sess.run([cost, lr, vgg.prob, true_out, merge_op],
                                             feed_dict={images: X, true_out: Y, train_mode: True})
            print "loss at " + str(step) + " {:.5f}".format(loss) + " lr: {:.5f}".format(lr_val)
            prob_val = np.average(prob_val, 0);
            gt_val = np.average(gt_val, 0);
            print prob_val
            print gt_val
            summary_writer.add_summary(merge, step)
        if (step % 2000 == 0):
            vgg.save_npy(sess, str(step) + '.npy')
    # test classification
  #  prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    # test classification again, should have a higher probability about tiger
   # prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
   # utils.print_prob(prob[0], './synset.txt')
    # test save
    #vgg.save_npy(sess, './test-save.npy')
