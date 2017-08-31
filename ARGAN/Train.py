import argparse
import numpy
import numpy as np
numpy.random.seed(1)
import tensorflow as tf
tf.set_random_seed(1)
import sys
from random import shuffle
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
from Model import Model
from Discriminator import Discriminator
from Input import BatchImageInput
import Losses
import time

#global config
config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
try:
  tf.Session(config=config)
except: pass


parser = argparse.ArgumentParser(description='Expert denoisers training')

parser.add_argument('--lr1', type=float, metavar='N', default=1e-5, required=True, help='learning rate is required')
parser.add_argument('--lr2', type=float, metavar='N', default=1e-5, required=True, help='learning rate is required')
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

ave_out = sys.stdout
if args.log_file is not None:
  f = open(os.path.join(LGXY,args.log_file),'a')
  sys.stdout = f


with tf.Session(config=config) as sess:
  step_ = tf.Variable(0, trainable=False)
  #lr_ = tf.train.exponential_decay(args.lr, step_, 100, 0.999,staircase=True)
  #tf.summary.scalar("lr", lr_)
  tf.summary.scalar("step", step_)
  optimizer1 = tf.train.AdamOptimizer(args.lr1)
  optimizer2 = tf.train.GradientDescentOptimizer(args.lr2)

  # model
  E = Model(sess,optimizer=optimizer1, batch_size = args.batch_size, tot_res_blocks=5)
  D = Discriminator(sess,optimizer=optimizer2, batch_size = args.batch_size)

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

#  z_ = E.inference(x_, phase_train)
  t_ = D.inference(y_, phase_train) # clean sample logits
  f_ = D.inference(x_, phase_train) # reconstructed sample logits

  prob_t_ = tf.reduce_mean(tf.nn.sigmoid(t_))
  prob_f_ = tf.reduce_mean(tf.nn.sigmoid(f_))

  Loss_list = {
    'lp':(Losses.lp_loss,0.9),
    'pl':(Losses.perceptual_loss,0.1),
    # 'tv':Losses.tv_loss,
  }

  Loss_ops = {}
  Backprop_ops = {}
  LOSS = 0
  for loss_name in Loss_list:
    loss_name
    # compute the loss
    Loss_ops[loss_name] = Loss_list[loss_name][0](y_,x_)
    # weighted sum
    LOSS = LOSS+ (Loss_ops[loss_name]*Loss_list[loss_name][1])
    # add o summary
    tf.summary.scalar(loss_name+'_loss', Loss_ops[loss_name])

  Loss_ops['total']  = LOSS
  tf.summary.scalar('loss_recon', LOSS)

  def return_random (min_val , max_val):
    # return tf.random_uniform([1], minval = min_val , maxval = max_val , dtype = tf.float32)
    return tf.constant(1.0)

  pos_loss = -tf.reduce_mean((tf.constant(1.0)*tf.log(prob_t_))) #+ ((1.0 - return_random(0.8,1.2))*tf.log(1.0-prob_t_)))
  neg_loss = -tf.reduce_mean((1.0 - tf.constant(0.0))*tf.log(1.0-prob_f_))

  disc_true_loss = pos_loss
  disc_false_loss = neg_loss 
  disc_loss = pos_loss + neg_loss
  Loss_ops['pos_loss'] = pos_loss
  Loss_ops['neg_loss'] = neg_loss
  Loss_ops['disc_loss'] = disc_loss
  Loss_ops['disc_true_loss']= disc_true_loss
  Loss_ops['disc_false_loss'] = disc_false_loss
 

  BACK_PROP_OP_adv = E.get_backprop(
                       genr_loss,
                       step_
                     )

  BACK_PROP_OP_dis = D.get_backprop(
                       disc_loss,
                       step_
                     )
  BACK_PROP_OP_true_dis = D.get_backprop(
                       disc_true_loss, 
 		 	step_
                       )
  BACK_PROP_OP_false_dis = D.get_backprop(
  		       disc_false_loss,
  			step_
     			)		

#  tf.summary.image("predk_image",z_)
  tf.summary.image("clean_image",y_)
  tf.summary.image("noisy_image",x_)
  

  # DATA FLOW

  X_train_,Y_train_ = Train_Data.get_minibatch_tensors()
  X_valdn_,Y_valdn_ = Valdn_Data.get_minibatch_tensors()
  t_coord   = tf.train.Coordinator()
  t_threads = tf.train.start_queue_runners(coord=t_coord)

'''If you want to save all the variables and visualize them in tensorboard then un comment this'''
  ## Add histograms for trainable variables.
#  for var in tf.trainable_variables():
#    tf.summary.histogram(var.op.name, var)
#  for v_,g_ in GRADS_:
#    if v_ is not None:
#      tf.summary.histogram(v_.op.name,g_)

  summary_op = tf.summary.merge_all()

  writer = tf.summary.FileWriter(os.path.join(LGXY, 'LOG'), graph=E.sess.graph)
  init   = tf.global_variables_initializer()
 # E.sess.run(init)
  D.sess.run(init)
 # E.load_model('../Exp000906/E2/checkpoints/E2.100000.ckpt')

  step   = args.start_step
  stop_step = args.tot_grad_updates - args.start_step
  i=0
  j=0
  while step <= stop_step:
    X_train_batch, Y_train_batch = E.sess.run([X_train_,Y_train_])

    summary = E.sess.run(
                [summary_op],
                feed_dict={x_:X_train_batch,y_:Y_train_batch, phase_train:True}
              )
  
    prob_t, prob_f, pos_l,neg_l,total_disc_loss,tl,fl = D.sess.run(
        [prob_t_,prob_f_,pos_loss,neg_loss,disc_loss,t_,f_],
        feed_dict={
          x_:X_train_batch,
          y_:Y_train_batch,
          phase_train:True
	}
      )
    if( True ):

     print step
     print "{} = {}".format("Total disc loss",total_disc_loss)
     print "{} = {}".format("Positive loss",pos_l)
     print "{} = {}".format("Negative loss",neg_l)
     print "{} = {} {} = {}".format("Prob_t",prob_t,"Prob_f",prob_f)
     print "{} = {}".format("True logit",np.mean(tl))
     print "{} = {}".format("False logit",np.mean(fl))
     print "{} = {}".format("Mean of x train batch",np.mean(X_train_batch))
     print "{} = {}".format("Mean of y train batch",np.mean(Y_train_batch))
     print "{} = {}".format("Mean squarred error between actual pair",((Y_train_batch - X_train_batch)**2).mean(axis  =None))
     print "************************"  
        
    if(False):
     sys.exit(0)

    j=j+1
    _ = D.sess.run(
            [BACK_PROP_OP_dis],
            feed_dict={
              x_:X_train_batch,
              y_:Y_train_batch,
              phase_train:True
            }
          )


    writer.add_summary(summary[0], step)

    if step % args.log_interval == 0 or step == stop_step:

      X_valdn_batch, Y_valdn_batch = D.sess.run([X_valdn_,Y_valdn_])
      

      # getting all the losses
      prob_tv, prob_fv, pos_lv,neg_lv,total_disc_lossv = D.sess.run(
        [prob_t_,prob_f_,pos_loss,neg_loss,disc_loss],
        feed_dict={
          x_:X_train_batch,
          y_:Y_train_batch,
          phase_train:False
        }
      )

      LOSSES = {}
      for loss_op in Loss_ops:
        op = Loss_ops[loss_op]
        out= D.sess.run(
               [op],
               feed_dict={
                 x_:X_train_batch,
                 y_:Y_train_batch,
                 phase_train:True
               }
             )
        LOSSES[loss_op+'_train'] = out[0]
        out= D.sess.run(
               [op],
               feed_dict={
                 x_:X_valdn_batch,
                 y_:Y_valdn_batch,
                 phase_train:True
               }
             )
        LOSSES[loss_op+'_valdn'] = out[0]
     
# Printing differrent losses
      loss_str_list = []
      for key in LOSSES:
          if  key=='disc_loss_train' or key=='disc_true_loss_train' or key=='disc_false_loss_train':

           loss_str_list.append(" {} = {:e}".format(key,LOSSES[key]))
  
      prob_t, prob_f = D.sess.run([prob_t_, prob_f_], feed_dict = {x_: X_valdn_batch, y_:Y_valdn_batch, phase_train:False})
  

    sys.stdout.flush()
    if step % args.snapshot_interval == 0 or step == stop_step:
      E.save_model(os.path.join(LGXY, 'checkpoints', "{}.{:05d}.ckpt".format(LGXY,step)))

    step += 1

  print("Done")



try:
  f.close()
except:
  pass
sys.stdout = _save_out
