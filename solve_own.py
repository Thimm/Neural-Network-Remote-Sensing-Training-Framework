#!/usr/bin/env python
# Martin Kersner, 2016/01/13

from __future__ import division
import sys
import os
import caffe
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import pickle
import Image
import scipy.misc
#np.set_printoptions(threshold=np.nan)
    

base_weights = '/space/export/data/zwie_th/DLR/NeuralNetworks/dilation/pretrained/dilation_front_Bachelor.caffemodel'

solver = caffe.SGDSolver('/space/export/data/zwie_th/DLR/NeuralNetworks/dilation/Solver/solver_B.prototxt')


# copy base weights for fine-tuning
solver.net.copy_from(base_weights)
caffe.set_mode_gpu()
caffe.set_device(0)
whichOne = 'conv1_1_B'
iterations = 100000
filtersold = None
open('final.txt', 'w')
with open('thesame.txt', 'w') as thefile:
	for i in xrange(iterations):
		solver.step(1)
#		net = solver.net
#		label = net.blobs['label'].data[0,:,:]
#		data = net.blobs['data'].data[0,0:3,:,:].transpose(1,2,0)
#		print data.shape
#                scipy.misc.imsave('label.png', label)
#                scipy.misc.imsave('data.jpg', data)
#                plt.figure(0)
#		plt.imshow(data.astype('int'))
#                
#		plt.figure(1)
#		plt.imshow(label)
#		plt.show()
#                filters = net.params[whichOne][0].data
#                print filters
#                print filters.shape
#        	try:
#			result = filtersold - filters
#			filtersold = filters
##			with open('final.txt', 'a') as final:
##				filters = filters[(filters!=0)]	
##				final.write('%s\n' % filters)
#		except:
#			filtersold = filters
#			pass
#			print ('Error happend')
	#print data
	#print data.shape
	
