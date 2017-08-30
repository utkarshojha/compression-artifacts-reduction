from tensorflow.python.ops import math_ops
import tensorflow as tf
import numpy

def lp_loss(y_true, y_pred):
  diff_sq = tf.squared_difference(y_true,y_pred)
  loss = tf.reduce_mean(diff_sq, axis=None)
  return loss



def tv_loss(y_true,y_pred):
  def total_variation(images):
    pixel_dif1 = images[:, 1:, :, :] - images[:, :-1, :, :]
    pixel_dif2 = images[:, :, 1:, :] - images[:, :, :-1, :]
    sum_axis = [1, 2, 3]
    tot_var = math_ops.reduce_sum(math_ops.abs(pixel_dif1), axis=sum_axis) + \
              math_ops.reduce_sum(math_ops.abs(pixel_dif2), axis=sum_axis)
    return tot_var

  loss = tf.reduce_mean(total_variation(y_pred))
  return loss



def perceptual_loss(y_true,y_pred, layer_list=['conv5_4'], alphas=[1.0]):
  def conv_layer(bottom, weight, bias=None, s=1, padding='SAME', relu=True, group=1):
    if group==1:
      conv = tf.nn.conv2d(bottom, weight, [1, s, s, 1], padding=padding)
    else:
      input_split = tf.split(bottom, group, 3)
      weight_split = tf.split(weight, group, 3)
      conv_1 = tf.nn.conv2d(input_split[0], weight_split[0], [1, s, s, 1], padding=padding)
      conv_2 = tf.nn.conv2d(input_split[1], weight_split[1], [1, s, s, 1], padding=padding)
      conv = tf.concat([conv_1, conv_2], 3)
    if bias is None:
      if relu:
        return tf.nn.relu(conv)
      else:
        return conv
    else:
      bias = tf.nn.bias_add(conv, bias)
      if relu:
        return tf.nn.relu(bias)
      else:
        return bias
  def max_pool(bottom, k=3, s=1, padding='SAME'):
    return tf.nn.max_pool(bottom, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=padding)
  def vgg_19(image, weights, biases, keep_prob=1.0):
    activations = {}
    shapes = image.get_shape().as_list()
    if shapes[3] == 1:
       rgb = tf.concat(axis = 3, values=[image,image,image])
    else: 
       rgb = image

    rgb_scaled = rgb*255.0
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
    VGG_MEAN = [103.939, 116.779, 123.68]
    bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
          ])

    image = bgr


    with tf.name_scope("conv1"):
      activations['conv1_1'] = conv_layer(image, weights['conv1_1'], biases['conv1_1'])
      activations['conv1_2'] = conv_layer(activations['conv1_1'], weights['conv1_2'], biases['conv1_2'])
      activations['pool1'] = max_pool(activations['conv1_2'], k=2, s=2)
      
    with tf.name_scope("conv2"):
      activations['conv2_1'] = conv_layer(activations['pool1'], weights['conv2_1'], biases['conv2_1'])
      activations['conv2_2'] = conv_layer(activations['conv2_1'], weights['conv2_2'], biases['conv2_2'])
      activations['pool2'] = max_pool(activations['conv2_2'], k=2, s=2)
      
    with tf.name_scope("conv3"):
      activations['conv3_1'] = conv_layer(activations['pool2'], weights['conv3_1'], biases['conv3_1'])
      activations['conv3_2'] = conv_layer(activations['conv3_1'], weights['conv3_2'], biases['conv3_2'])
      activations['conv3_3'] = conv_layer(activations['conv3_2'], weights['conv3_3'], biases['conv3_3'])
      activations['conv3_4'] = conv_layer(activations['conv3_3'], weights['conv3_4'], biases['conv3_3'])
      activations['pool3'] = max_pool(activations['conv3_4'], k=2, s=2)
      
    with tf.name_scope("conv4"):
      activations['conv4_1'] = conv_layer(activations['pool3'], weights['conv4_1'], biases['conv4_1'])
      activations['conv4_2'] = conv_layer(activations['conv4_1'], weights['conv4_2'], biases['conv4_2'])
      activations['conv4_3'] = conv_layer(activations['conv4_2'], weights['conv4_3'], biases['conv4_3'])
      activations['conv4_4'] = conv_layer(activations['conv4_3'], weights['conv4_4'], biases['conv4_4'])
      activations['pool4'] = max_pool(activations['conv4_4'], k=2, s=2)
      
    with tf.name_scope("conv5"):
      activations['conv5_1'] = conv_layer(activations['pool4'], weights['conv5_1'], biases['conv5_1'])
      activations['conv5_2'] = conv_layer(activations['conv5_1'], weights['conv5_2'], biases['conv5_2'])
      activations['conv5_3'] = conv_layer(activations['conv5_2'], weights['conv5_3'], biases['conv5_3'])
      activations['conv5_4'] = conv_layer(activations['conv5_3'], weights['conv5_4'], biases['conv5_4'])
      activations['pool5'] = max_pool(activations['conv5_4'], k=2, s=2)
    return activations

  def get_weights():
    net = numpy.load('../Resources/VGG/vgg19.npy').item()
    weights = {}
    biases  = {}
    for name in net.keys():
      weights[name] = tf.Variable(tf.constant(net[name][0]), dtype='float32' ,name=name+"_weight", trainable=False)
      biases[name]  = tf.Variable(tf.constant(net[name][1]), dtype='float32' ,name=name+"_bias", trainable=False)
    return weights,biases

  weights,biases = get_weights()

  y_true_activations = vgg_19(y_true,weights,biases)
  y_pred_activations = vgg_19(y_pred,weights,biases)
  
  for layer_name in layer_list:
    assert layer_name in y_true_activations.keys()

  loss = 0
  for layer_name,alpha in zip(layer_list,alphas):
    y = y_true_activations[layer_name]/255.0
    x = y_pred_activations[layer_name]/255.0
    loss += alpha* lp_loss(y,x)
  return loss


def cross_entropy_sigmoid(logits,labels):
  loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits,labels = labels)
  return tf.reduce_mean(loss)
