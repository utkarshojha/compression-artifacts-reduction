import os
import glob
import tensorflow as tf

import random
from tqdm import tqdm
from itertools import product

class BatchImageInput():
  def __init__(self, file_paths, noise_levels, crop_size = [64, 64, 3], batch_size = 32):
    self.batch_size   = batch_size
    self.noise_levels = noise_levels
    self.crop_size    = crop_size
    self.file_paths = file_paths
  
  def read_one_file(self, filename_queue):
    print filename_queue
    image_reader = tf.WholeFileReader()
    _,I   = image_reader.read(filename_queue)
    Id = tf.image.decode_image(I, channels=3)
    Id = tf.to_float(Id)
    Ic = tf.random_crop(Id, size = self.crop_size)
    
    tt = tf.cast(Ic,tf.uint8)
    tt = tf.image.encode_jpeg(tt,quality=20) # have to remove this hardcoding later
    In = tf.image.decode_jpeg(tt, channels=3)
    In = tf.reshape(In, self.crop_size)
    In = tf.to_float(In)
    
    return Ic,In

  def get_minibatch_tensors(self, num_epochs=None):
    input_queue = tf.train.string_input_producer(self.file_paths, num_epochs=num_epochs, shuffle=True)
    noises_ = tf.constant(self.noise_levels)
    shuff_noises_ = tf.random_shuffle(noises_)

    Ic, In = self.read_one_file(input_queue)
    # adding noise to a clean image
    # In = Ic + tf.random_normal(Ic.get_shape(), mean=0.0, stddev=shuff_noises_[0], dtype=tf.float32)
    
    Ic_batch, In_batch = tf.train.batch([Ic, In], batch_size=self.batch_size, capacity = 100)
    # (X,Y)
    return In_batch/255.0, Ic_batch/255.0

if __name__ == '__main__':
  with open('imagenet_val_paths.txt', 'r') as f:
    file_paths = f.read().splitlines()

  EI = BatchImageInput(file_paths,[10.0,15.0],batch_size = 32)
  
  x_,y_ = EI.get_minibatch_tensors()
  
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  print "haha"
  with tf.Session(config=config) as sess:
    tf.global_variables_initializer()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    x,y = sess.run([x_,y_])
    print x[0][:10,:10,0],y[0][:10,:10,0]
    print x[0].shape
    coord.request_stop()
    coord.join(threads)
  
