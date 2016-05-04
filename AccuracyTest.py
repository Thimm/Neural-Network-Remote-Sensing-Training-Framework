import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import xdibias
import scipy.misc
import os
def open_label(path):
   '''
   opens the xdibias file at path location
   '''
   info  = xdibias.Image(path)
   im = info.readImageData()
   im = im.transpose([1,0])[::-1]
   return (im)

def get_image(pathLabel, pathGt):
   '''
   gets the image and label path
   '''
   paths = []
   for subdir, dirs, files in os.walk(pathLabel):
      for file in files:
         la = os.path.join(pathLabel, file)
         gt = os.path.join(pathGt, file[:-10]+'.roof')
         if os.path.exists(gt):
            paths.append([la, gt])
   print paths
   return paths


def total_acc(l):
   '''
   calculates the average accuracy of all images. Adding all intsects together and dividing these by 
   all unions would probably have been the better option!!!
   '''
   acc_total = sum(l) / float(len(l))
   return acc_total

def get_acc(im_path, gt_path):
   '''
   dirty version of getting the accuracy of the images. Uses numpy slicing to get the ground truth  on the same size as the resulting segmentation maps
   '''
   gt = open_label(gt_path)
   im = Image.open(im_path)
   im = np.array(im)[:,:,2]
   scipy.misc.imsave('im.png',im)
   gt = gt[186:(3984+186),186:(2656+186)]
   scipy.misc.imsave('gt.png',gt)
   gt[gt > 0] = 1
   im[im > 0] = 1
   inter = (np.logical_and(gt,im)).astype('int')
   union = np.logical_or(gt, im).astype('int')

   acc = float(np.sum(inter)) / float(np.sum(union))
   return acc

def main(path='/space/export/data/zwie_th/Files/eichenau_4k/Results/', '/space/export/data/zwie_th/Files/eichenau_4k/groundtruth'):
   '''
   writes accruacy of each result to a file and calculates the average over all of them
   '''
   acc = []
   paths = get_image(path)
   with open('Accuracy_Eichenau.txt', 'w') as f:
      for path in paths:
         a = (get_acc(path[0], path[1]))
         print 'Accuracy for: ', path[0][-15:], a
         acc.append(a)
         f.write('Accuracy for ' + path[0][-15:] + ':   ' + str(a) + '\n')


   print 'Average Acc:, ', total_acc(acc)
   with open('Accuracy_Eichenau.txt', 'a') as f:
      f.write('Average of all accuracys ' + str(total_acc(acc)))
   return

if __name__ == '__main__':
    main()
