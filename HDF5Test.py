import os, math, sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import h5py

rootdir = '/space/export/data/zwie_th/DLR/Bachelorarbeit/HDF5_Test'
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith('.h5'):
            f = h5py.File(os.path.join(subdir, file), 'r')
            data = f['/data']
            label = np.array(f['/label'], dtype=np.int)
            newdata = np.empty((900,900, 4), dtype=np.float32)
            mean_pixel = np.array([102.93, 111.36, 116.52, 116.52], dtype=np.float32)
            newdata[...] =  data[()].transpose([2,3,1,0])[:,:,:,0]
           # plt.figure(0)
           # plt.imshow(newdata.astype('int')[:,:,2], cmap='gray')
           # im = Image.fromarray(newdata[:,:,2::-1].astype('int'))
           # im.save('TestData.jpg')
           # plt.figure(1)
           # plt.imshow(label[0,:,:])
           ## plt.show()i
            print newdata 
            raw_input("Press Enter to continue...")


