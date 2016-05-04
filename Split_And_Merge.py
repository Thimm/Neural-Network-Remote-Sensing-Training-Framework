import os, math, sys
from osgeo import gdal
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import h5py
from osgeo import osr
import caffe
import cv2
import subprocess
from re import split
import cPickle as pickle
from scipy.misc import imsave
def clip(input_List, raster, path='tiles/', slice_size=1700):
   
   y_off = int(input_List[0][1][0])
   x_off = int(input_List[0][0][0])
   newPath = path + str(input_List[1][0]) + '_' + str(input_List[2][0]) + '.tif'
   print newPath, x_off, y_off
   cmd = ('gdal_translate -srcwin %s' %x_off+' %s'%y_off+' %s'%slice_size+' %s'%slice_size+' '
          '%s'%raster+' -of GTiff ' '%s'%newPath)
   subprocess.call(cmd, shell=True)
   return (newPath)


def get_image_info(dataFile):
    """ gets image and finds all kind of information as well as returns a n-dimensional np array
    normalizes imgae between 0 and 1
    Usually uses GDAL to get Inforamtion of the data, but in the case of an erro
    lsit will try to open the file with the help of xdibias. Maybe there is a better Version
    but this was the easiest one"""
       
    dsI = gdal.Open(dataFile)

    sizeX = dsI.RasterXSize 
    sizeY = dsI.RasterYSize 
    rCount = dsI.RasterCount

    return {'xSize':sizeX, 'ySize':sizeY, 'rasterCount':rCount}

def preperation(image_shape, slice_size=1700, margin=186):
   '''
   creates a List of coordinates for the tiles of an input Image 
   + The position where each tile is located
   slice_size = size of the actual output
   margin =  will be subtracted from the slice_size. Is the actual output of the neuralnetwork

   returns a list with the image coordinates and its position (0, 0) in the new image
   '''
   tiles = []
   pos1, pos2, x1 = 0, 0, 0

   h, w = image_shape['xSize'], image_shape['ySize']
   print image_shape, h, w
   out_size = slice_size - 2*margin
   nr_CR = ((h-margin*2) // out_size, (w-margin*2) // out_size)

   for x in xrange(margin+out_size/2, w - slice_size/2, out_size):
       pos1 += 1
       for y in xrange(margin+out_size/2, h - slice_size/2, out_size):           
           if x != x1: pos2 = 0
           pos2 += 1         
           
           # old: image = [(y-(slice_size/2)), (y+(slice_size/2))],\
           #               [(x-(slice_size/2)), (x+(slice_size/2))]]
           image = [[y-(slice_size/2)], [x-(slice_size/2)]]
           tiles.append([image, [pos1], [pos2]])
           x1 = x
   print tiles
   return tiles

def image_open(f):
    dsI = gdal.Open(f)
    rCount = dsI.RasterCount
    im = np.float32(dsI.GetRasterBand(1).ReadAsArray())
    if dsI.RasterCount > 1:
       for (bandnew) in range(2, (rCount+1)):
           new = np.float32(dsI.GetRasterBand(bandnew).ReadAsArray())      
           im = np.dstack((im, new))
    mean_pixel = np.array([102.93, 111.36, 116.52, 116.52], dtype=np.float32)
    im = im*255 - mean_pixel
    return im

def predict(model_path, pretrained, data):
    caffe.set_mode_gpu()
    caffe.set_device(1)
    net = caffe.Net(model_path, pretrained, caffe.TEST)
    zoom = 8
    input_dims = net.blobs['data'].shape
    batch_size, num_channels, input_height, input_width = input_dims
    caffe_in = np.zeros(input_dims, dtype=np.float32)
    caffe_in[0] = data.transpose([2,0,1])
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    prob = out['prob'][0]
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

def for_dilation(slice_size, margin, h, w):
    out_size = slice_size - 2*margin
    nr_CR = ((h-margin*2) // out_size, (w-margin*2) // out_size)
    return out_size, nr_CR


def main(dataFile='/space/export/data/zwie_th/DLR/TrainingData/Bachelorarbeit_Data/PleiadesData/combined1.tif', pretrained='/space/export/data/zwie_th/DLR/EichenauTraining/Snapshots/Dilation/20160502/Bachelor_base_lr_12_iter_1628.caffemodel', model_path='/space/export/data/zwie_th/DLR/NeuralNetworks/dilation/models/Bachelorarbeit/dilated_convolution_front_B_Pre.prototxt', slice_size=1700, margin=186):
    path = 'tiles/'
    info = get_image_info(dataFile)
    params = for_dilation(slice_size, margin, info['xSize'], info['ySize'])
    # get image info
    
    #for preperation of the tiles
    #tiles = preperation(info, slice_size , margin)

    #for prediciton
    paths = []
    #for tile in tiles:
    # clip
    #    paths.append(clip(tile, dataFile, path))
    tiles = []
    for i in os.listdir(path):
       f = image_open(path+i)
       s = split(r'[_.]+', i)
       tiles.append([predict(model_path, pretrained, f), (int(s[0]), int(s[1]))])

    try:
        f = open('tiles_dump','wb') 
        pickle.dump(tiles, f)
    except:
        pass
    im = join_t(tiles, params[1], params[0])
    print im
    try:
        imsave('merged_together.png', im)
    except:
        pass
    try:
        im.save('merged_together_PIL.png')
    except:
        pass
        

if __name__ == '__main__':
    main()
      
    
