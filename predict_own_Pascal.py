#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import caffe
import cv2
import numpy as np
from os.path import dirname, exists, join, splitext
import sys

__author__ = 'Fisher Yu'
__copyright__ = 'Copyright (c) 2016, Fisher Yu'
__email__ = 'i@yf.io'
__license__ = 'MIT'


PALLETE = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128]
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
    f = h5py.File(os.path.join(subdir, file), 'r')
    data = f['/data']
    net = caffe.Net(model_path, pretrained, caffe.TEST)
    zoom = 8
    input_dims = net.blobs['data'].shape
    batch_size, num_channels, input_height, input_width = input_dims
    caffe_in = np.zeros(input_dims, dtype=np.float32)
    caffe_in[0] = data 
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    prob = out['prob'][0]
    image_size = (prob.shape[1]*zoom, prob.shape[2]*zoom, 3)
    zoom_prob = interp_map(prob, zoom, prob.shape[1]*zoom,prob.shape[2]*zoom)
    prediction = np.argmax(zoom_prob.transpose([1, 2, 0]), axis=2)
    color_image = PALLETE[prediction.ravel()]
    np.save('nparray', zoom_prob.transpose([1, 2, 0]), True)
    color_image = color_image.reshape(image_size)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, color_image)


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
    pretrained = '/space/export/data/zwie_th/DLR/NeuralNetworks/dilation/models/Bachelorarbeit/dilated_convolution_front_B_Pre.prototxt' 
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
    output_path = splitext(args.input_path)[0] + suffix
    predict(model_path, pretrained,
            args.input_path, output_path)


if __name__ == '__main__':
    main()
