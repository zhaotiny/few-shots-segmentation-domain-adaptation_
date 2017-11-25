import tensorflow as tf
import os
import numpy as np
import re
## leaky relu
def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

## define the layer of fully-connected + batch normalization + nonlinearity
def dense_batch_relu(x, num_out, phase, scope, alpha = 0.01):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, num_out,
                                               activation_fn=None,
                                               scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1,
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn', updates_collections=None)
        return lrelu(h2, alpha)

 ## define fully-conneted layer
## define the fully connected layer
def dense(x, num_out, scope):
    with tf.variable_scope(scope):
        return tf.contrib.layers.fully_connected(x, num_out,
                                             activation_fn=None,
                                             scope='dense')

## regression network that maps small smaples model to large sample models
def regress_net(x, phase, num_input, n_hidden_1, n_hidden_2, n_hidden_3):
    fc1 = dense_batch_relu(x, n_hidden_1, phase, "fc1")
    fc2 = dense_batch_relu(fc1, n_hidden_2, phase, "fc2")
    fc3 = dense_batch_relu(fc2, n_hidden_3, phase, "fc3")
    fc4 = dense(fc3, num_input, "fc4")
    fc4 = tf.nn.sigmoid(fc4)
    return fc4
## define conv2d with our own weights and features
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME', data_format = "NCHW")
    x = tf.nn.bias_add(x, b, data_format = "NCHW")
    return x
## conv the models with the features (output from the regression network) to obtain the prediction
def prediction_net(F, y_pred):
    W = y_pred[:, 0: 576]
    B = y_pred[:, 576]
    W = tf.transpose(W, (1, 0))
    W = tf.reshape(W, (64, 3, 3, -1))
    W = tf.transpose(W, (1, 2, 0, 3))
    pred = conv2d(F, W, B)
    return pred
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
## save the snapshot to as .npy format
def snapshot_npy(sess, output_dir, name, iter):
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
    filename = output_dir + '/' + name + str(iter) + '.npy'
    np.save(filename,data);
    print 'Wrote snapshot to: {:s}'.format(filename)
# get variable given scope/name/weight:0
def get_variable(scope, name, weights):
    tmp = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope + '/' + name + '/' + weights + ':0')
    return tmp[0]

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

## compute the segmentaion loss
def cal_loss(logits, labels, NUM_CLASSES):
    if (NUM_CLASSES == 6):
        loss_weight = np.array([
            0.2533,
            3.1525,
            3.4525,
            21.3407,
            32.5793,
            0.1050]) # class 0~5
    else:
        loss_weight = np.array([
            1.1702,
            0.2092,
            6.5827,
            1.3543,
            3.4586,
            0.0603,
            0.2511,
            0.1356,
            1.0074,
            2.1081,
            7.0991,
            1.0246,
            2.2594,
            1.7446,
            11.1466,
            13.0023,
            0.2856,
            0.8580,
            0.1342,
            5.4837,
            7.6757,
            0.7505,
            1.1470,
            0.8595,
            3.1807,
            0.9113,
            7.8915,
            7.2087])
    labels = tf.cast(labels, tf.int32)
    # return loss(logits, labels)
    return weighted_loss(logits, labels, num_classes=NUM_CLASSES, head=loss_weight)

## visualize the prediction output
def visualize(predicted, NUM_CLASSES):
    if (predicted.shape[1] == 1):
        ind = np.squeeze(predicted[0,:,:,:])
    else:
        output = np.squeeze(predicted[0, :, :, :])
        ind = np.argmax(output, axis=0)
    r = ind.copy()
    g = ind.copy()
    b = ind.copy()
    if NUM_CLASSES == 6:
        building = [70,	70,	70]
        Road = [128,64,128]
        Car = [64,0,128]
        Pedestrian = [64,64,0]
        Bicyclist = [0,128,192]
        Unlabelled = [0,0,0]

        label_colours = np.array([Road, Car, building, Pedestrian, Bicyclist, Unlabelled])
    else:
        unlabeled = [0, 0, 0]
        ego_vehicle = [0, 0, 0]
        static = [20, 20, 20]
        dynamic = [111,	74,	0]
        ground = [81, 0, 81]
        road = [128, 64, 128]
        sidewalk = [244, 35, 232]
        building = [70,	70,	70]
        wall = [102, 102, 156]
        fence = [190, 153, 153]
        guard_rail = [80, 165, 180]
        bridge = [150, 100, 100]
        tunnel = [150, 120, 90]
        pole = [153, 153, 153]
        traffic_light = [250, 170, 30]
        traffic_sign = [220, 220, 0]
        vegetation = [107, 142, 35]
        terrain = [152, 251, 152]
        sky = [70, 130, 180]
        person = [220, 20, 60]
        rider = [255, 0, 0]
        car = [0, 0, 142]
        truck = [0, 0, 70]
        bus = [0, 60, 100]
        trailer = [0, 0, 110]
        train = [0, 80, 100]
        motorcycle = [0, 0, 230]
        bicycle = [119, 11, 32]
        label_colours = np.array([unlabeled, ego_vehicle, static, dynamic, ground, road, sidewalk, building, wall, \
                         fence, guard_rail, bridge, tunnel, pole, traffic_light, traffic_sign, vegetation, \
                         terrain, sky, person, rider, car, truck, bus, trailer, train, motorcycle, bicycle])

    for l in range(0,NUM_CLASSES):
        r[ind==l] = label_colours[l,0]
        g[ind==l] = label_colours[l,1]
        b[ind==l] = label_colours[l,2]

    rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
    rgb[:,:,0] = r/255.0
    rgb[:,:,1] = g/255.0
    rgb[:,:,2] = b/255.0

    rgb = rgb.astype(np.float32, copy=False)
    return rgb

## define the layer of standard conv
def sparse_conv(x, num_outputs, kernel_size, phase, scope):
    with tf.variable_scope(scope):
        return tf.contrib.layers.conv2d(x, num_outputs, kernel_size,
                                padding='SAME', data_format = "NCHW", scope = "sparse")
## get train variables given scope
def get_train_var(scope):
    all_variables_trained = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    num_var = len(all_variables_trained)
    conv_variables = []
    for i in range(num_var):
        variable_name = all_variables_trained[i].name
        if (re.search(scope, variable_name) != None):
            conv_variables.append(all_variables_trained[i])
            continue
    return conv_variables
#define regularization loss
def regularization_loss(w_pred_denorm, conv_W, lambda1):
    W = w_pred_denorm[:, 0: 576]
   # B = w_pred_denorm[:, 576]
    W = tf.transpose(W, (1, 0))
    W = tf.reshape(W, (64, 3, 3, -1))
    W = tf.transpose(W, (1, 2, 0, 3))
    reg_loss = lambda1 * tf.nn.l2_loss(conv_W - W);
    #oss = tf.reduce_mean(loss + beta * regularizer)
    return reg_loss

# convert caffe conv filter to tensorflow filter
def conv_caffe_to_tf(convW, NUM_CLASSES):
    convW = np.reshape(convW, (NUM_CLASSES, -1))
    convW = np.transpose(convW, (1, 0))

    convW = np.reshape(convW, (64, 3, 3, -1))
    convW = np.transpose(convW, (1, 2, 0, 3))
    return convW
