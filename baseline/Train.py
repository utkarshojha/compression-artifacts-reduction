import argparse
import numpy
import numpy as np
numpy.random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)

from random import shuffle
import os
import glob
from PIL import Image

from Model import Model
from Input import BatchImageInput
import Losses
import time

#global config
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
try:
  tf.Session(config=config)
except: pass


parser = argparse.ArgumentParser(description='Expert denoisers training')

parser.add_argument('--lr', type=float, metavar='N', default=1e-5, required=True, help='learning rate is required')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', required=True, help='input batch size for training (default: 32)')
parser.add_argument('--tot-grad-updates', type=int, default=10000, metavar='N', help='No of iterations')
parser.add_argument('--start-step', type=int, default=1, metavar='N', help='training start step')
parser.add_argument('--log-file', type=str, default=None, metavar='N', help='No of iterations')
parser.add_argument('--train-dir', type=str, default=None, metavar='N', help='No of iterations')
parser.add_argument('--valdn-dir', type=str, default=None, metavar='N', help='No of iterations')
parser.add_argument('--log-interval', type=int, default=10000, metavar='N', required=True, help='frequency of logging')
parser.add_argument('--snapshot-interval', type=int, default=10000, metavar='N', required=True, help='frequency of snapshotting')
parser.add_argument('--exp-code', type=str, default=10000, metavar='N', required=True, help='experiment code')

args = parser.parse_args()

LGXY = "{}".format(args.exp_code)

def make_dirs(dirs):
  for _dir in dirs:
    if not os.path.isdir(_dir):
      os.mkdir(_dir)

make_dirs([LGXY, os.path.join(LGXY, 'LOG'), os.path.join(LGXY, 'checkpoints')])


# redirect error messages to std out or file
import sys
_save_out = sys.stdout
if args.log_file is not None:
  f = open(os.path.join(LGXY,args.log_file),'a')
  sys.stdout = f

with tf.Session(config=config) as sess:
  step_ = tf.Variable(0, trainable=False)
  lr_ = tf.train.exponential_decay(args.lr, step_, 100, 0.999,staircase=True)
  tf.summary.scalar("lr", lr_)
  tf.summary.scalar("step", step_)
  optimizer = tf.train.AdamOptimizer(lr_)

  # model
  E = Model(sess,optimizer=optimizer, batch_size = args.batch_size, tot_res_blocks=5)

  # --beg-- train data
  with open('../Resources/imagenet_val_paths.txt', 'r') as f:
    file_paths = [os.path.join(args.train_dir,fname) for fname in f.read().splitlines()]
  Train_Data = BatchImageInput(file_paths,[20],batch_size = args.batch_size)

  # --beg-- validation data
  with open('../Resources/bsds500_test_paths.txt', 'r') as f:
    file_paths = [os.path.join(args.valdn_dir,fname) for fname in f.read().splitlines()]
  Valdn_Data = BatchImageInput(file_paths,[20],batch_size = args.batch_size)
  # --end-- validation data

  # computation
  x_ = tf.placeholder(tf.float32,[args.batch_size,64,64,3],name="noisy_images")
  y_ = tf.placeholder(tf.float32,[args.batch_size,64,64,3],name="clean_images")
  a_ = tf.placeholder(tf.float32,[], name="alpha")
  phase_train = tf.placeholder(tf.bool, [], name='is_training')

  z_      = E.inference(x_, phase_train)
  Loss_list = {
    'lp':(Losses.lp_loss,0.9),
    'pl':(Losses.perceptual_loss,0.1),
    # 'tv':Losses.tv_loss,
  }

  Loss_ops = {}
  Backprop_ops = {}
  LOSS = 0
  for loss_name in Loss_list:
    print loss_name
    # compute the loss
    Loss_ops[loss_name] = Loss_list[loss_name][0](y_,z_)
    # weighted sum
    LOSS = LOSS+ (Loss_ops[loss_name]*Loss_list[loss_name][1])
    # add o summary
    tf.summary.scalar(loss_name+'_loss', Loss_ops[loss_name])

  Loss_ops['total']  = LOSS
  tf.summary.scalar('total_loss', LOSS)

  BACK_PROP_OP = E.get_backprop(
                   LOSS,
                   step_
                 )

  tf.summary.image("predk_image",z_)
  tf.summary.image("clean_image",y_)
  tf.summary.image("noisy_image",x_)
  

  # DATA FLOW

  X_train_,Y_train_ = Train_Data.get_minibatch_tensors()
  X_valdn_,Y_valdn_ = Valdn_Data.get_minibatch_tensors()
  t_coord   = tf.train.Coordinator()
  t_threads = tf.train.start_queue_runners(coord=t_coord)


  ## Add histograms for trainable variables.
  #for var in tf.trainable_variables():
  #  tf.summary.histogram(var.op.name, var)
  #for v_,g_ in GRADS_:
  #  if v_ is not None:
  #    tf.summary.histogram(v_.op.name,g_)

  summary_op = tf.summary.merge_all()

  writer = tf.summary.FileWriter(os.path.join(LGXY, 'LOG'), graph=E.sess.graph)
  init   = tf.global_variables_initializer()
  E.sess.run(init)

  step   = args.start_step
  stop_step = args.tot_grad_updates - args.start_step

  while step <= stop_step:
    # get train batch
    X_train_batch, Y_train_batch = E.sess.run([X_train_,Y_train_])

    # get op summary
    summary = E.sess.run(
                [summary_op],
                feed_dict={x_:X_train_batch,y_:Y_train_batch, phase_train:False}
              )

    # run one grad update
    _ = E.sess.run(
          [BACK_PROP_OP],
          feed_dict={
            x_:X_train_batch,
            y_:Y_train_batch,
            phase_train:True
          }
        )
    writer.add_summary(summary[0], step)


    if step % args.log_interval == 0 or step == stop_step:

      X_valdn_batch, Y_valdn_batch = E.sess.run([X_valdn_,Y_valdn_])

      # getting all the losses
      LOSSES = {}
      for loss_op in Loss_ops:
        op = Loss_ops[loss_op]
        out= E.sess.run(
               [op],
               feed_dict={
                 x_:X_train_batch,
                 y_:Y_train_batch,
                 phase_train:False
               }
             )
        LOSSES[loss_op+'_train'] = out[0]
        out= E.sess.run(
               [op],
               feed_dict={
                 x_:X_valdn_batch,
                 y_:Y_valdn_batch,
                 phase_train:False
               }
             )
        LOSSES[loss_op+'_valdn'] = out[0]


      # printing all losses
      loss_str_list = []
      for key in LOSSES:
        loss_str_list.append(" {} = {:e}".format(key,LOSSES[key]))
      print("[{}]  Step = {:5d} {}".format(time.strftime("%A:%B-%d-%Y %H:%M:%S"),step, "".join(loss_str_list)))

    sys.stdout.flush()
    if step % args.snapshot_interval == 0 or step == stop_step:
      E.save_model(os.path.join(LGXY, 'checkpoints', "{}.{:05d}.ckpt".format(LGXY,step)))

    step += 1

  print("Done")
  #tf.reset_default_graph()


try:
  f.close()
except:
  pass
sys.stdout = _save_out
