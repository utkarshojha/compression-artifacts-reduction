from __future__ import print_function

import tensorflow as tf
import re


TOWER_NAME ='tower'
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32# if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev=5e-2):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float32# if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  return var



def batch_norm(inputs, is_training, pop_mean, pop_var, beta, gamma, decay = 0.999):
  def train():
    batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2], name='moments')
    train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    train_var  = tf.assign(pop_var , pop_var  * decay + batch_var * (1 - decay))
    with tf.control_dependencies([train_mean, train_var]):
      return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, variance_epsilon=1e-3)
  
  def testing():
    return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, variance_epsilon=1e-3)

  return tf.cond(is_training, train, testing)
    





class Model(object):
  def __init__(self,sess,optimizer, batch_size=32, tot_res_blocks=2):
    self.sess = sess
    self.optimizer = optimizer
    self.batch_size = batch_size
    self.THETA = {}
    self.tot_res_blocks = tot_res_blocks
    self.init_model()
    self.saver = tf.train.Saver(self.THETA, max_to_keep=100)


  def get_training(self):
    return self.training

  def set_training(self, training):
    self.training = tf.assign(self.training, tf.constant(training, shape=[]))
  
  def init_model(self):
    def init_bn_params(block_name,ind,size=64):
      self.THETA['{}beta{}'.format(block_name,ind)] = tf.Variable(
                      tf.truncated_normal([size], mean=0.0, stddev=5e-2), name='{}beta{}'.format(block_name,ind), trainable=True)
      self.THETA['{}gamma{}'.format(block_name,ind)]= tf.Variable(
                      tf.truncated_normal([size], mean=1.0, stddev=5e-2), name='{}gamma{}'.format(block_name,ind), trainable=True)
      self.THETA['{}pop_mean{}'.format(block_name,ind)] = tf.Variable(tf.zeros([size]), name = '{}pop_mean{}'.format(block_name,ind), trainable=False)
      self.THETA['{}pop_var{}'.format(block_name,ind)]  = tf.Variable(tf.ones([size]) , name = '{}pop_var{}'.format(block_name,ind) , trainable=False)
 
    self.THETA['inpw1'] = _variable_with_weight_decay(shape=[3, 3, 3, 64], name='inw1')
    self.THETA['inpb1']    = _variable_with_weight_decay(shape=[ 64], name='inb1')

    for ind in range(1,self.tot_res_blocks+1):

        self.THETA['b{}w1'.format(ind)] = _variable_with_weight_decay(shape=[3, 3, 64, 64], name='b{}w1'.format(ind))
        self.THETA['b{}b1'.format(ind)] = _variable_with_weight_decay(shape=[ 64]         , name='b{}b1'.format(ind))
       
        init_bn_params('b{}'.format(ind),1)

        self.THETA['b{}w2'.format(ind)] = _variable_with_weight_decay(shape=[3, 3, 64, 64], name='b{}w2'.format(ind))
        self.THETA['b{}b2'.format(ind)] = _variable_with_weight_decay(shape=[ 64]         , name='b{}b2'.format(ind))

        init_bn_params('b{}'.format(ind),2)
        

    self.THETA['recon0w1']  = _variable_with_weight_decay(shape=[3, 3,  64,  64], name='recon0w1')
    self.THETA['recon0b1']  = _variable_with_weight_decay(shape=[64]            , name='recon0b1')
    init_bn_params('recon0',1)

    self.THETA['recon1w1']  = _variable_with_weight_decay(shape=[3, 3,  64, 256], name='recon1w1')
    self.THETA['recon1b1']  = _variable_with_weight_decay(shape=[256]           , name='recon1b1')
    
    self.THETA['recon2w1']  = _variable_with_weight_decay(shape=[3, 3, 256, 256], name='recon2w1')
    self.THETA['recon2b1']  = _variable_with_weight_decay(shape=[256]           , name='recon2b1')
    
    self.THETA['recon3w1']  = _variable_with_weight_decay(shape=[3, 3, 256,   3], name='recon3w1')
    self.THETA['recon3b1']  = _variable_with_weight_decay(shape=[3]             , name='recon3b1')




  def inference(self,inp, phase_train):

    def conv(blk,ind,inp):
      x = tf.nn.conv2d(inp, self.THETA['{}w{}'.format(blk,ind)], strides=[1, 1, 1, 1], padding='SAME',name="{}w{}".format(blk,ind))
      x = tf.nn.bias_add(x, self.THETA['{}b{}'.format(blk,ind)], name="{}b{}".format(blk,ind))
      return x

    def bn(blk,ind,inp):
      x = batch_norm(inp, 
            phase_train, 
            self.THETA['{}pop_mean{}'.format(blk,ind)],
            self.THETA['{}pop_var{}'.format(blk,ind)],
            self.THETA['{}beta{}'.format(blk,ind)],
            self.THETA['{}gamma{}'.format(blk,ind)],
          )
      return x
      

    def block_conv(inp, name):
      with tf.variable_scope(name):
        x  = conv(name,1,inp)
        x  = bn(name,1,x)
        c1 = tf.nn.relu(x)
        
        x  = conv(name,2,c1)
        x  = bn(name,2,x)

        x = tf.add(inp,x)
      return x


    x = conv('inp',1,inp)
    o0 = tf.nn.relu(x)
    o = o0

    # all residual blocks
    for ind in range(1,self.tot_res_blocks+1):
      o = block_conv(o,"b{}".format(ind))

    x = conv('recon0',1,o)
    x = bn('recon0',1,x)
    x = tf.add(x,o0)

    x = conv('recon1',1,x)
    x = tf.nn.relu(x)

    x = conv('recon2',1,x)
    x = tf.nn.relu(x)

    x = conv('recon3',1,x)

    out = x
    return out


  def save_model(self,file_name):
      self.saver.save(self.sess,file_name)

  def load_model(self,file_name):
      self.saver.restore(self.sess, file_name)

  
  def get_gradients(self,total_loss):
    grads = self.optimizer.compute_gradients(total_loss)

    return grads

  def get_backprop(self,total_loss,global_step):
    solver = self.optimizer.minimize(total_loss,global_step=global_step)
    return solver
