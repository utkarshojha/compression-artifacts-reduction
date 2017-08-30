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
from skimage import measure
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
#  print "Session started"
  step_ = tf.Variable(0, trainable=False)
  #lr_ = tf.train.exponential_decay(args.lr, step_, 100, 0.999,staircase=True)
  #tf.summary.scalar("lr", lr_)
  tf.summary.scalar("step", step_)
  optimizer1 = tf.train.GradientDescentOptimizer(learning_rate = args.lr1 )
  optimizer2 = tf.train.GradientDescentOptimizer(learning_rate = args.lr2 )
#  print "Otimizers set"
  # model
  E = Model(sess,optimizer=optimizer1, batch_size = args.batch_size, tot_res_blocks=5)
  D = Discriminator(sess,optimizer=optimizer2, batch_size = args.batch_size)
#  print "Model set"
  # --beg-- train data
  with open('../Resources/imagenet_val_paths.txt', 'r') as f:
    file_paths = [os.path.join(args.train_dir,fname) for fname in f.read().splitlines()]
  Train_Data = BatchImageInput(file_paths,[50],batch_size = args.batch_size)

  # --beg-- validation data
  with open('../Resources/bsds500_test_paths.txt', 'r') as f:
    file_paths = [os.path.join(args.valdn_dir,fname) for fname in f.read().splitlines()]
  Valdn_Data = BatchImageInput(file_paths,[50],batch_size = args.batch_size)
  # --end-- validation data
#  print "Data files loaded"
  # computation
#  init   = tf.global_variables_initializer()
#  E.sess.run(init)
#  D.sess.run(init)
#  E.load_model('../Exp000906/E2/checkpoints/E2.100000.ckpt')


  x_ = tf.placeholder(tf.float32,[args.batch_size,64,64,3],name="noisy_images")
  y_ = tf.placeholder(tf.float32,[args.batch_size,64,64,3],name="clean_images")
  a_ = tf.placeholder(tf.float32,[], name="alpha")
  phase_train = tf.placeholder(tf.bool, [], name='is_training')
#  print "Placeholders set"
 # z_ = E.inference(x_, phase_train)
  t_ = D.inference(y_, phase_train) # clean sample logits
  f_ = D.inference(x_, phase_train) # reconstructed sample logits
#  print "Inferences calculated"
  prob_t_ = tf.reduce_mean(tf.nn.sigmoid(t_))
  prob_f_ = tf.reduce_mean(tf.nn.sigmoid(f_))
#  print "Probabilities done"
  Loss_list = {
    'lp':(Losses.lp_loss,0.9),
    'pl':(Losses.perceptual_loss,0.1),
    # 'tv':Losses.tv_loss,
  }
#  print "LOss list done"
  Loss_ops = {}
  Backprop_ops = {}
  LOSS = 0
  for loss_name in Loss_list:
    print "hi"+str(loss_name)
    # compute the loss
    Loss_ops[loss_name] = Loss_list[loss_name][0](y_,x_)
    # weighted sum
    LOSS = LOSS+ (Loss_ops[loss_name]*Loss_list[loss_name][1])
    # add o summary
    tf.summary.scalar(loss_name+'_loss', Loss_ops[loss_name])

  Loss_ops['total']  = LOSS
  tf.summary.scalar('loss_recon', LOSS)

  def return_random (min_val , max_val):
     return tf.random_uniform([1], minval = min_val , maxval = max_val , dtype = tf.float32)
#    if(min_val == 0.0 and max_val == 0.2):
#     return tf.constant(0.0)
#    else:
#     return tf.constant(1.0)
#  pos_loss = Losses.cross_entropy_sigmoid(logits = t_,labels = tf.random_uniform([1],minval=0.7,maxval=1.2,dtype=tf.float32)*tf.ones_like(t_))
#  neg_loss = Losses.cross_entropy_sigmoid(logits = f_,labels = tf.random_uniform([1],minval=0.0,maxval=0.3,dtype=tf.float32)*tf.ones_like(f_))
#  adv_loss = Losses.cross_entropy_sigmoid(logits = f_,labels = tf.random_uniform([1],minval=0.7,maxval=1.2,dtype=tf.float32)*tf.ones_like(f_))
  '''Modification begins here'''
#  print "Positive loss and negative loss computation"
  pos_loss = -tf.reduce_mean(tf.log(prob_t_))
  neg_loss = -tf.reduce_mean(tf.log(1.0 - prob_f_))
#pos#  pos_loss = -tf.reduce_mean((return_random(0.9,1.1)*tf.log(prob_t_)) + ((1.0 - return_random(0.9,1.1))*tf.log(1.0-prob_t_)))
#neg#  neg_loss = -tf.reduce_mean((return_random(0.0,0.2)*tf.log(prob_f_)) + (1.0 - return_random(0.0,0.2))*tf.log(1.0-prob_f_))
#  smoothen_high = tf.random_uniform([1], minval=0.7 , maxval= 1.2, dtype=tf.float32)
#  smoothen_low = tf.random_uniform( [1], minval=0.0, maxval= 0.3, dtype = tf.float32)
#  disc_loss = -tf.reduce_mean((tf.random_uniform([1], minval=0.7 , maxval= 1.2, dtype=tf.float32)*tf.log(prob_t_)) + ((1.0 - (tf.random_uniform( [1], minval=0.0, maxval= 0.3, dtype = tf.float32)))*tf.log(1 - prob_f_)))
#  genr_loss = -tf.reduce_mean(smoothen_high*tf.log(prob_f_))
#  disc_loss = -tf.reduce_mean(tf.log(prob_t_) + tf.log(1 - prob_f_))
#gener#  gener_loss = -tf.reduce_mean(tf.log(prob_f_))
#gen#  gener_loss = -tf.reduce_mean((return_random(0.9,1.1)*tf.log(prob_f_)) + ((1.0 - return_random(0.9,1.1))*tf.log(1.0 - prob_f_)))
#  print "Adversaraial loss computation done"
#  disc_true_loss = -tf.reduce_mean(tf.log(prob_t_))
#  disc_false_loss = -tf.reduce_mean(tf.log(1 - prob_f_))
#LOSS#  LOSS = LOSS
#genr_loss#  genr_loss  = (gener_loss) + (10.0*LOSS) 
#  print "Generator loss computation done"
  '''End here'''
  disc_true_loss = pos_loss
  disc_false_loss = neg_loss 
  disc_loss = pos_loss + neg_loss
#  print "All losses done"
#  genr_loss = adv_loss # + (0.8*LOSS) 
  Loss_ops['pos_loss'] = pos_loss
  Loss_ops['neg_loss'] = neg_loss
#  Loss_ops['adv_loss'] = adv_loss
  Loss_ops['disc_loss'] = disc_loss
#  Loss_ops['genr_loss'] = genr_loss
  Loss_ops['disc_true_loss']= disc_true_loss
  Loss_ops['disc_false_loss'] = disc_false_loss
#  print "ALl losses saved to the dictionary"
 # alph = 0.1 
 # LOSS_adv = (1-alph)* LOSS + (alph)* adv_loss
#  LOSS_adv = genr_loss
#  Loss_ops['model_loss'] = LOSS_adv
  

#  tf.summary.scalar('loss_adv', genr_loss)

#  BACK_PROP_OP_adv = E.get_backprop(
#                       genr_loss,
#                       step_
#                     )

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
#  print "BAckpropagation code done"
#  tf.summary.image("predk_image",z_)
  tf.summary.image("clean_image",y_)
  tf.summary.image("noisy_image",x_)
  
#  print "Summary done"

  # DATA FLOW

  X_train_,Y_train_ = Train_Data.get_minibatch_tensors()
  X_valdn_,Y_valdn_ = Valdn_Data.get_minibatch_tensors()
  t_coord   = tf.train.Coordinator()
  t_threads = tf.train.start_queue_runners(coord=t_coord)
#  print "Mnibatch tensors retrieved"

  ## Add histograms for trainable variables.
#  for var in tf.trainable_variables():
#    tf.summary.histogram(var.op.name, var)
#  for v_,g_ in GRADS_:
#    if v_ is not None:
#      tf.summary.histogram(v_.op.name,g_)

  summary_op = tf.summary.merge_all()

  writer = tf.summary.FileWriter(os.path.join(LGXY, 'LOG'), graph=E.sess.graph)
  init   = tf.global_variables_initializer()
  E.sess.run(init)
  D.sess.run(init)
 # print "Initializing the session"
  E.load_model('../Exp000906/E2/checkpoints/E2.100000.ckpt')
#  print "Model loaded"
  step   = args.start_step
  stop_step = args.tot_grad_updates - args.start_step
  i=0
  j=0
#  print "Now enterin the main loop"
  while step <= stop_step:
#    print  "Getting the training batch"
    X_train_batch, Y_train_batch = E.sess.run([X_train_,Y_train_])
#    print "IN THE LOOP: Batches extracted"
#    print Y_train_batch.shape

#    print Y_train_batch[0][:,:,0]
#    print "Getting the noisy batch" 
#    X_train_batch = sess.run(tf.random_uniform([32,64,64,3], minval = 0.0 , maxval = 1.0 , dtype = tf.float32))
#    print "*********************"
#    print X_train_batch.shape
#    print X_train_batch[0][:,:,0]
   # sys.exit(0)
    # get op summary
    
    
    summary = E.sess.run(
                [summary_op],
                feed_dict={x_:X_train_batch,y_:Y_train_batch, phase_train:False}
              )
#    print "Session run for summary operation"
#    print "Training probabilities"
    
    prob_t, prob_f, pos_l,neg_l,total_disc_loss,tl,fl= D.sess.run(
        [prob_t_,prob_f_,pos_loss,neg_loss,disc_loss,t_,f_],
        feed_dict={
          x_:X_train_batch,
          y_:Y_train_batch,
          phase_train:False
	}
      )
#    print "{} = {}".format("Generator loss after first SESSION RUN",generator_loss)
#    print "{} = {}".format("Generator loss after second SESSION RUN",E.sess.run(genr_loss,feed_dict = { x_:X_train_batch , y_ : Y_train_batch,phase_train:False}))  
#    print "All the variables extracted by running the session"
    y = Y_train_batch
    x = X_train_batch
#    z[ z > 1.0 ] = 1.0
#    z[ z < 0.0 ] = 0.0
#    print "Small name conversion"
#    np.save('x'+str(step)+'.npy',x)
#    np.save('y'+str(step)+'.npy',y)

#    z = predk = E.sess.run(z_,feed_dict={x_:x, phase_train:False})
#    np.save('z'+str(step)+'.npy',z)
 
#    if(step ==2):
#      sys.exit(0)    
#    np.save('y'+str(step)+'.npy',y)
#    np.save('x'+str(step)+'.npy',x)
#    np.save('z'+str(step)+'.npy',z)
#    if (step == 5):
#       sys.exit(0)
      
#    z = (z-np.min(z))/(np.max(z) - np.min(z))
#    print "Entering the FALSE if statement"
  #  if(False):
  #   for i in xrange(3):
 
   #   print "***************THIS IS Y BAND "+str(i)+"********************"
   #   print y[1][:,:,i].shape
   #   print y[1][:,:,i] 
   #   print "***************THIS IS Z BAND "+str(i)+"********************"
   #   print z[1][:,:,i].shape
   #   print z[1][:,:,i]
   #   print "{} = {}".format("MSE between Y and Z",(np.abs((y[1][:,:,i]-z[1][:,:,i]))).mean(axis = None))
   #   print "{} = {}".format("PSNR between Y and Z",measure.compare_psnr(y[1][:,:,i],z[1][:,:,i],dynamic_range = 1.0))    
    if( True ):
#      print "{} = {}".format("This is the clean image batch",Y_train_batch[0])
#      print "{} = {}".format("This is the batch not used",X_train[0])
#      print "{} = {}".format("This is the noisy batch created",X_train_batch[0])  
#    print "blah blah extracted by running the session"
#    assert total_disc_loss == pos_l + neg_l
##     print "{} = {}".format("Input shape",x[0].shape)
##     print x[0]
##     print "{} = {}".format("Output shape",z[0].shape)
##     print z[0]
##     print "{} = {}".format("MSE of input and output",((z[0]-x[0])**2).mean(axis = None))
##     print "{} = {}".format("PSNR of input and output",measure.compare_psnr(z[0],x[0],dynamic_range = 1.0)) 
     print step
     #print "{} = {}".format("Shape of the reconstructed image",z.shape)
  ##   print "{} = {}".format("MSE of clean and reconstructed batch",((Y_train_batch[0,:,:,:] - z[0,:,:,:])**2).mean(axis = None))
  ##   print "{} = {}".format("MSE of clean and noisy images",((Y_train_batch[0,:,:,:] - X_train_batch[0,:,:,:])**2).mean(axis = None))
  ##   print "{} = {} {} {}".format("Max value of Y and X and Z",np.max(Y_train_batch[0,:,:,:]),np.max(X_train_batch[0,:,:,:]),np.max(z[0,:,:,:]))
###     print "{} = {}".format("Mean of the clean image band 1",np.mean(y[:,:,:,0]))
###     print "{} = {}".format("Mean of the clean image band 2",np.mean(y[:,:,:,1]))
###     print "{} = {}".format("Mean of the clean image band 3",np.mean(y[:,:,:,2]))
###     print "{} = {}".format("Mean of the noisy image band 1",np.mean(x[:,:,:,0]))
###     print "{} = {}".format("Mean of the noisy image band 2",np.mean(x[:,:,:,1]))
###     print "{} = {}".format("Mean of the noisy image band 3",np.mean(x[:,:,:,2])) 
###     print "{} = {}".format("Mean of the reconstructed image band 1",np.mean(z[:,:,:,0]))
###     print "{} = {}".format("Mean of the reconstructed image band 2",np.mean(z[:,:,:,1]))
###     print "{} = {}".format("Mean of the reconstructed image band 3",np.mean(z[:,:,:,2]))

#     print "*******************CLEAN IMAGE************************"
#     print y[0,:,:,0]
#     print "**********************"
#     print y[0,:,:,1]
#     print "**********************"
#     print y[0,:,:,2]
#     print "*******************NOISY IMAGE************************"
#     print x[0,:,:,0]
#     print "**********************"
#     print x[0,:,:,1]
#     print "**********************"
#     print x[0,:,:,2] 
#     print "*******************RECONSTRUCTED IMAGE************************"
#     print z[0,:,:,0]
#     print "**********************"
#     print z[0,:,:,1]
#     print "**********************"
#     print z[0,:,:,2] 
#     print "***************************DONE******************************"
##     print "{} = {} , {} = {}".format("MSE of clean and noisy (0)",((y[0,:,:,0] - x[0,:,:,0])**2).mean(axis = None),"clean and reconstructed",((y[0,:,:,0] - z[0,:,:,0])**2).mean(axis = None))
##     print "{} = {} , {} = {}".format("MSE of clean and noisy (1)",((y[0,:,:,1] - x[0,:,:,1])**2).mean(axis = None),"clean and reconstructed",((y[0,:,:,1] - z[0,:,:,1])**2).mean(axis = None))
##     print "{} = {} , {} = {}".format("MSE of clean and noisy (2)",((y[0,:,:,2] - x[0,:,:,2])**2).mean(axis = None),"clean and reconstructed",((y[0,:,:,2] - z[0,:,:,2])**2).mean(axis = None))
   #  psnr_r = []
   #  psnr_n = []
   #  mse_r = []
   #  mse_n = []
   #  for i in xrange(32):
   #    psnr_r.append(measure.compare_psnr(y[i] , z[i] , dynamic_range = 1.0))
   #    psnr_n.append(measure.compare_psnr(y[i] , x[i] , dynamic_range = 1.0))
   #    mse_r.append(np.mean((y[i] - z[i])**2))
   #    mse_n.append(np.mean((y[i] - x[i])**2))
   #  psnr_r = np.mean(np.asarray(psnr_r))
   #  psnr_n = np.mean(np.asarray(psnr_n))
   #  mse_r = np.mean(np.asarray(mse_r))
   #  mse_n = np.mean(np.asarray(mse_n))
   #  print "{} = {} : {} = {}".format("MSE of Y and X",mse_n,"ME of Y and Z",mse_r)
   #  print "{} = {}".format("Maximum of Y",np.max(y))
   #  print "{} = {}".format("Maximum of X",np.max(x))
   #  print "{} = {}".format("Maximum of Z",np.max(z))
   #  print "{} = {} : {} = {}".format("PSNR of noisy image",psnr_n,"PSNR of reconstructed image",psnr_r)
#     sys.exit(0)
     print "*************************"
     print "{} = {}".format("Total disc loss",total_disc_loss)
     print "{} = {}".format("Positive loss",pos_l)
     print "{} = {}".format("Negative loss",neg_l)
   #  print "*************************"
   #  print "{} = {}".format("Adversarial loss",adv_loss)
   #  print "{} = {}".format("Content loss",con_loss)
   #  print "{} = {}".format("Total Generator loss",generator_loss)
     print "*************************"
     print "{} = {} {} = {}".format("Prob_t",prob_t,"Prob_f",prob_f)
     print "{} = {}".format("True logit",np.mean(tl))
     print "{} = {}".format("False logit",np.mean(fl))
     print "*************************"
##     print "{} = {}".format("MSE OF RECONSTRUCTED AND NOISY",((x-z)*2).mean(axis = None))
##     print "{} = {}".format("Shape of input",x.shape)
##     print "{} = {}".format("Shape of reconstructed image",z.shape)
   #  print "{} = {} {} = {}".format("Max value of Y",np.max(Y_train_batch),"Max value of X",np.max(X_train_batch)) 
#     print "{} - {}".format("PSNR of current reconstructed batch",measure.compare_psnr(Y_train_batch*255.0, X_train_batch*255.0,dynamic_range = 255.0))
#     print "{} = {}".format("Mean of x train batch",np.mean(X_train_batch))
#     print "{} = {}".format("Mean of y train batch",np.mean(Y_train_batch))
#     print "{} = {}".format("Mean squarred error between the noisy and clean batch",((Y_train_batch - X_train_batch)**2).mean(axis = None))
#     print "{} = {}".format("Mean squarred error between actual pair",((Y_train_batch - X_train_batch)**2).mean(axis  =None))
#    print "{} = {}".format("Mean of the not used one",np.mean(X_train))
     print "**********************************************************************************************"  
        
    if(False):
     sys.exit(0)
    # run one grad update 
   # if prob_t >0.9 or prob_f<0.1:
   # if (True): 
 
 #    i=i+1
   #  print "{} = {}".format("The adversarial loss backpropagated",E.sess.run(genr_loss,feed_dict={
   #       x_:X_train_batch,
   #       y_:Y_train_batch,
   #       phase_train:False
   #     }))

   #  _ = E.sess.run(
   #         [BACK_PROP_OP_adv],
   #         feed_dict={
   #           x_:X_train_batch,
   #           y_:Y_train_batch,
   #           phase_train:True
   #         }
   #       )
#    print "{} = {}".format("The adversarial loss backpropagated",D.sess.run(genr_loss,feed_dict={
#          x_:X_train_batch,
#          y_:Y_train_batch,
#          phase_train:False
#        }
#))
   # if prob_t <0.9 or prob_f>0.1:
    print "{} = {}".format("The discriminator loss backpropagated",E.sess.run(disc_loss,feed_dict={
          x_:X_train_batch,
          y_:Y_train_batch,
          phase_train:False
        }
))
  #  if(True):
    j=j+1
    _ = D.sess.run(
            [BACK_PROP_OP_dis],
            feed_dict={
              x_:X_train_batch,
              y_:Y_train_batch,
              phase_train:True
            }
          )
#    print "{} = {}".format("The discriminator loss backpropagated",E.sess.run(disc_loss,feed_dict={
#          x_:X_train_batch,
#          y_:Y_train_batch,
#          phase_train:False
#        }
#))

#    _ = D.sess.run(
#            [BACK_PROP_OP_false_dis],
#            feed_dict={
#              x_:X_train_batch,
#              y_:Y_train_batch,
#              phase_train:True
#            }
#          )

  #  _= D.sess.run(
  #	    [BACK_PROP_OP_false_dis],
  #	   feed_dict = {
  #             x_:X_train_batch,
  #	       y_:Y_train_batch,
  #	       phase_train:True
  #            }
  #     )				

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
#      assert pos_lv + neg_lv == total_disc_lossv 
#      print "Validation disc posotive loss "+str(pos_lv)
#      print "Validation disc negative loss "+str(neg_lv)
#      print "Validation disc total loss "+str(total_disc_lossv)
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
     
#      print "****************************"
#      print "Training and valiation loss" 

      # printing all losses
      loss_str_list = []
      for key in LOSSES:
          if  key=='disc_loss_train' or key=='disc_true_loss_train' or key=='disc_false_loss_train':

           loss_str_list.append(" {} = {:e}".format(key,LOSSES[key]))
#      print("[{}]Step={:5d}{}".format(time.strftime("%A:%B-%d-%Y %H:%M:%S"),step, "".join(loss_str_list)))
  
      prob_t, prob_f = D.sess.run([prob_t_, prob_f_], feed_dict = {x_: X_valdn_batch, y_:Y_valdn_batch, phase_train:False})
#      print "***************************"
#      print "Probabilities"
#      print "Prob_true "+str(prob_t)+" Prob_false "+str(prob_f)
      
#    print i,j
    sys.stdout.flush()
   # if step % args.snapshot_interval == 0 or step == stop_step:
   #   E.save_model(os.path.join(LGXY, 'checkpoints', "{}.{:05d}.ckpt".format(LGXY,step)))

    step += 1

  print("Done")
  #tf.reset_default_graph()


try:
  f.close()
except:
  pass
sys.stdout = _save_out
