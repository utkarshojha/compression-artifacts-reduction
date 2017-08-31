import argparse
parser = argparse.ArgumentParser(description='Expert denoisers training')
parser.add_argument('--image-filename', type=str, default='', metavar='N', required=True, help='File containing PSNR values of LIVE1 dataset')
args = parser.parse_args()

import sys


noisy_psnr = []
recon_psnr = []
diff_psnr = []
i=0
with open(args.image_filename,'r') as f:
  for l in f.readlines():
    i=i+1
    l = l.strip()
    l = l.split()
    noisy_psnr.append(float(l[1]))
    recon_psnr.append(float(l[2]))
    diff_psnr.append( float(l[2]) - float(l[1]))
   

avg_noisy_psnr = sum(noisy_psnr)/len(noisy_psnr)
avg_recon_psnr = sum(recon_psnr)/len(recon_psnr)
avg_diff_psnr = sum(diff_psnr)/len(diff_psnr)
print "Average noisy PSNR = "+str(avg_noisy_psnr)
print "Average recon PSNR = "+str(avg_recon_psnr)
print "Average diffr PSNR = "+str(avg_diff_psnr) 

