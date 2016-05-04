#!/usr/bin/env python
# -*- coding: utf-8 -*-

from osgeo import gdal
from osgeo import osr
import argparse
import caffe
import cv2
import numpy as np
from os.path import dirname, exists, join, splitext
import sys
import h5py
import os
import matplotlib.pyplot as plt
import Image
import osgeo
__author__ = 'Fisher Yu'
__copyright__ = 'Copyright (c) 2016, Fisher Yu'
__email__ = 'i@yf.io'
__license__ = 'MIT'


PALLETE = np.array([
    [0, 0, 0],
    [128, 0, 0],
], dtype=np.uint8)



def interp_map(prob, zoom, width, height):
    zoom_prob = np.zeros((prob.shape[0], height, width), dtype=np.float32)
    for c in range(prob.shape[0]):
        for h in range(height):
            for w in range(width):
                r0 = h // zoom
                r1 = r0 + 0
                c0 = w // zoom
                c1 = c0 + 0
                rt = float(h) / zoom - r0
                ct = float(w) / zoom - c0
                v0 = rt * prob[c, r1, c0] + (1 - rt) * prob[c, r0, c0]
                v1 = rt * prob[c, r1, c1] + (1 - rt) * prob[c, r0, c1]
                zoom_prob[c, h, w] = (1 - ct) * v0 + ct * v1
    return zoom_prob


def predict(model_path, pretrained, input_path, output_path):
    f = h5py.File(input_path, 'r')
    data = f['/data']
    net = caffe.Net(model_path, pretrained, caffe.TEST)
    zoom = 8
    input_dims = net.blobs['data'].shape
    batch_size, num_channels, input_height, input_width = input_dims
    caffe_in = np.zeros(input_dims, dtype=np.float32)
    caffe_in[0] = data 
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    prob = out['prob'][0].transpose([1,2,0])
    print (prob)
    image_size = (prob.shape[0]*zoom, prob.shape[1]*zoom)
    zoom_prob = cv2.resize(prob, image_size) 
    prediction = np.argmax(zoom_prob, axis=2)
    print (prediction)
    prediction [prediction==1] = 200
    np.save('nparray', zoom_prob, True)
    cv2.imwrite(output_path, prediction)

def getHDF5(path):
    f = h5py.File(path, 'r')
    data = f['/data']
    label = np.array(f['/label'], dtype=np.int)
    newdata = np.empty((900,900, 4), dtype=np.float32)
    mean_pixel = np.array([102.93, 111.36, 116.52, 116.52], dtype=np.float32)
    newdata[...] =  data[()].transpose([2,3,1,0])[:,:,:,0] + mean_pixel
    print newdata[:,:,1:4].shape
    im = Image.fromarray(np.uint8(newdata[:,:,1:4]))
    im.save('Orig.jpg', 'JPEG')
    dst_ds = gdal.GetDriverByName('GTiff').Create('myGeoTIFF.tif', newdata.shape[0], newdata.shape[1], newdata.shape[2], gdal.GDT_Byte)
    print newdata
    dst_ds.GetRasterBand(2).WriteArray(newdata[:,:,0])   # write r-band to the raster
    dst_ds.GetRasterBand(1).WriteArray(newdata[:,:,1])   # write g-band to the raster
    dst_ds.GetRasterBand(0).WriteArray(newdata[:,:,2])
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None
    return

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
    pretrained = '/space/export/data/zwie_th/DLR/EichenauTraining/Snapshots/Dilation/20160502/Bachelor_iter_600.caffemodel' 
    if not exists(pretrained):
        raise sys.exit('Error: Run pretrained/download.sh first to '
                       'download pretrained model weights')
    if args.context:
        suffix = '_context.png'
        model_path = join(dirname(__file__), 'models',
                          'dilated_convolution_front_own_pre.prototxt')
    else:
        suffix = '_front.png'
        model_path = '/space/export/data/zwie_th/DLR/NeuralNetworks/dilation/models/Bachelorarbeit/dilated_convolution_front_B_Pre.prototxt'
    output_path = splitext(args.input_path)[0] + suffix
    predict(model_path, pretrained,
            args.input_path, output_path)
    #getHDF5(args.input_path)

if __name__ == '__main__':
    main()
