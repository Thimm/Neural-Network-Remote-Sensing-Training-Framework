import os, math, sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from osgeo import osr
import caffe
import cv2
from random import shuffle
from math import sqrt, ceil, floor
import argparse
from os.path import dirname, exists, join, splitext
import cPickle as pickle
import scipy.misc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', nargs='?', default='',
                        help='Required path to input image')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID to run CAFFE. If -1, CPU is used, '
                             'which is the default setting.')
    parser.add_argument('--context', type=int, default=0,
                        help='Use context module')
    args = parser.parse_args()
    if args.input_path == '':
        sys.exit('Error: No path to input image')
    if not exists(args.input_path):
        sys.exit("Error: Can't find input image " + args.input_path)
    if args.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
        print('Using GPU ', args.gpu)
    else:
        caffe.set_mode_cpu()
        print('Using CPU')
    pretrained = '/space/export/data/zwie_th/DLR/EichenauTraining/Snapshots/Dilation/20160421/own_iter_6826.caffemodel' 
    if not exists(pretrained):
        raise sys.exit('Error: Run pretrained/download.sh first to '
                       'download pretrained model weights')
    if args.context:
        suffix = '_context.png'
        model_path = join(dirname(__file__), 'models',
                          'dilated_convolution_front_own_pre.prototxt')
    else:
        suffix = '_front.png'
        model_path = join(dirname(__file__), 'models',
                          'dilated_convolution_front_own_pre.prototxt')
    for subdir, dirs, files in os.walk(args.input_path):
        for file in files:
            if file.endswith('JPG'):
               
               print os.path.join(subdir, file[:-4]) + suffix
               output_path = os.path.join(subdir, 'Results', file[:-4]) + suffix
 #   output_path = args.input_path[:-4] + suffix
               print os.path.join(subdir, file)
               prep(model_path, pretrained,
                     os.path.join(subdir, file), output_path)
               
                  

def prep(model_path, pretrained, input_path, output_path):
   tiles = []
   pos1, pos2, x1 = 0, 0, 0
   slice_size = 1700
   margin = 186
   print 'input_path:', input_path
   image = cv2.imread(input_path, 1)
   h, w, d = image.shape
   out_size = slice_size - 2*margin
   nr_CR = ((h-margin*2) // out_size, (w-margin*2) // out_size)
   print margin+out_size/2
   for x in range(margin+out_size/2, w - slice_size/2, out_size):
       pos1 += 1
       for y in range(margin+out_size/2, h - slice_size/2, out_size):           
           print x, y
           if x != x1: pos2 = 0
           pos2 += 1         
           
           image1 = image[(y-(slice_size/2)):(y+(slice_size/2)),\
                          (x-(slice_size/2)):(x+(slice_size/2)), :]
           tiles.append([image1[:,:,::-1], (pos1, pos2)])
           print pos1, pos2, image1[:,:,::-1].shape
           x1 = x
   newTiles = []
   print len(tiles)
   for tile in tiles:
       print (type(tile), 'started')
       tile = (predict(model_path, pretrained, tile[0], output_path), tile[1])
       newTiles.append(tile)
   im = join_t(newTiles, nr_CR, out_size)
  
   scipy.misc.imsave(output_path, im)
   #cv2.imwrite(output_path, np.array(im))
   print output_path
#   plt.figure(0)
#   plt.imshow(im)
#   plt.axis('off')
#   plt.figure(1)
#   plt.imshow(image[:,:,::-1])
#   plt.axis('off')
#   plt.show()
   
   return

def predict(model_path, pretrained, input_path, output_path):
    net = caffe.Net(model_path, pretrained, caffe.TEST)
    zoom = 8
    mean_pixel = np.array([102.93, 111.36, 116.52], dtype=np.float32)
    input_dims = net.blobs['data'].shape
    batch_size, num_channels, input_height, input_width = input_dims
    caffe_in = np.zeros(input_dims, dtype=np.float32)
    image = input_path.astype(np.float32) - mean_pixel
    caffe_in[0] = image.transpose([2, 0, 1])
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    prob = out['prob'][0]
    image_size = (prob.shape[1]*zoom, prob.shape[2]*zoom, 3)
    zoom_prob = cv2.resize(prob.transpose([1,2,0]), (prob.shape[1]*zoom, prob.shape[2]*zoom))
    prediction = np.argmax(zoom_prob, axis=2)
    return (prediction)

def join_t(tiles, nr_CR, slice_size):
    """
    @param ``tiles`` - Tuple of ``Image`` instances.
    @return ``Image`` instance.
    """
    print nr_CR, 'nr_Cr'
    columns, rows = nr_CR
    print 'Image Size: ', slice_size*int(nr_CR[1]), slice_size*int(nr_CR[0])
    im = Image.new('RGB', (slice_size*int(nr_CR[1]), slice_size*int(nr_CR[0])), None)
    for tile in tiles:
        pos = (tile[1][0]*slice_size-slice_size, tile[1][1]*slice_size-slice_size)
        print tile[0]
        tile[0][tile[0]==1] = 155 
        data = Image.fromarray(np.uint8(tile[0])) 
        im.paste(data, (pos))
    return im


if __name__ == '__main__':
    main()
