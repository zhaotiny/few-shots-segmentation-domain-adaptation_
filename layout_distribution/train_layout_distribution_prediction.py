"""
train a network to predict the layout distribution
"""

import tensorflow as tf

import vgg19_trainable as vgg19
import pdb
from Dataset import Dataset
import os
import numpy as np
import re
NUM_CLASS = 5

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
true_out = tf.placeholder(tf.float32, [None, NUM_CLASS])
file_name = 'cityAll.txt'
display_step = 40
#init = tf.global_variables_initializer()

vgg = vgg19.Vgg19('vgg19.npy')
vgg.build(images)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = vgg.fc8, labels= true_out))

data = Dataset(file_name, num_classes=NUM_CLASS)

f = open(file_name, 'r')
output_file = []
for line in f:
    files = line.strip('\r\n').split()
    tmp = files[1]
    tmp = re.sub('freq', 'freq_pred', tmp)
    if (not os.path.exists(os.path.dirname(tmp))):
        os.mkdir(os.path.dirname(tmp))

    output_file.append(tmp)
f.close()

num_sample = data.total_size
batch_size = 1
num_Iter = num_sample / batch_size

avg_loss = 0.0
np.set_printoptions(precision=3, suppress=True)
count = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(0, num_sample):
        X, Y = data.next_batch(batch_size)
        [prob_val] = sess.run([vgg.prob], feed_dict={images: X})
        prov_val_tmp = np.zeros((1, NUM_CLASS))
        prov_val_tmp[0][0] = prob_val[0][0]
        prov_val_tmp[0][1] = prob_val[0][1] + prob_val[0][2]
        prov_val_tmp[0][2] = prob_val[0][3]
        prov_val_tmp[0][3] = prob_val[0][4]
        prov_val_tmp[0][4] = prob_val[0][5]

        if (step % display_step == 0):
            print prov_val_tmp
            print Y
            print
        freq = prov_val_tmp.squeeze(0)
        np.save(output_file[step], freq)

    # test classification
  #  prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    # test classification again, should have a higher probability about tiger
   # prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
   # utils.print_prob(prob[0], './synset.txt')
    # test save
