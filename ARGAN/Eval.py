import argparse
parser = argparse.ArgumentParser(description='Expert denoisers training')
parser.add_argument('--image-folder', type=str, default='', metavar='N', required=True, help='Image folder containing all the images')
parser.add_argument('--image-filename', type=str, default='', metavar='N', required=True, help='Image file name')
parser.add_argument('--quality', type=int, default='', metavar='N', required=True, help='quality')  
args = parser.parse_args()

import skimage.measure as measure
from PIL import Image
import numpy as np
import os
import sys
import math

quality = args.quality
image_folder = args.image_folder
clean_image_folder = os.path.join(image_folder , "clean_images")
noisy_image_folder = os.path.join(image_folder , "noise_std_" + str(quality))
recon_image_folder = os.path.join(image_folder , "recon_std_" + str(quality))

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# note: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1]* imageA.shape[2])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err



def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


image_file = args.image_filename
#print image_file
with open(image_file,'r') as f:
  for l in f.readlines():
    l = l.strip()
    l = l.rsplit('/',1)[1]

    Ic  = np.asarray(Image.open(os.path.join(clean_image_folder,l)), dtype = 'float32')

    In  = np.asarray(Image.open(os.path.join(noisy_image_folder,l)), dtype = 'float32') 
    Ir  = np.asarray(Image.open(os.path.join(recon_image_folder,l)), dtype = 'float32')

    print l+" "+str(psnr(Ic,In))+" "+str(psnr(Ic,Ir))
