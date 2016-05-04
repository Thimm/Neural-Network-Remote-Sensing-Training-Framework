import os, math
from sys.exc_traceback import tb_lineno 
from osgeo import gdal
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import xdibias
import lmdb
import caffe
import cv2
from random import shuffle
# Only needed for HDF5 creation
# import h5py

# How many degrees the cookie cutter should be rotated
degrees = 10
# The size which the squares are cliped in pixels
clipSize = 900
# How far the center points of the rotation should be 
stepSize = 500 
# Input Images. Special case for labelPath with Xdibias files
rasterPath =  '/space/export/data/zwie_th/Files/eichenau_4k'
labelPath =  rasterPath
# Where the output is
path_dst = '/space/export/data/zwie_th/DLR/TrainingData/dilation_all'
# prefix of the datasets that will be created
prefix = 'dataset'



def main(degrees, clipSize, stepSize, rasterpath, labelpath, HDF5, path_dst):  
    #gets driver information
    #loops through layer, gets angle from attributes and writes file with degree information  
    index = 0
    #this is important  because the programm uses not the distance from the middle to the side
    #but to the corners and the plus 2 are added, because afterwards it will get cropped again to the actual size the NN is using. Is only important for the coordinates creation
    clipSizeData = int(clipSize * math.cos(math.radians(45)))+2
    # needs to be 372 pixels smaller. because thats what the neural networks gives as an ouput
    margin = 186
    clipSizeLabel = clipSize - 2*margin
    dim = (clipSizeLabel/8, clipSizeLabel/8)
    # creates a fille list of both the raster data and the ground truth
    fileList = make_list(rasterpath, labelpath)

    #Iterates over the list
    for path in fileList['f']:
        # Loads the image infos, plus the Image
        imInfo0 = get_image_info(path[0])
        imInfo1 = get_image_info(path[1])        
        # allpoints holds all coordinates of the squares that will be used for cropping
        allpoints = calculates_coord(imInfo0['xSize'], imInfo0['ySize'], degrees, \
           clipSizeData, stepSize)
        # the squares get shuffled. Otherwise it would be needed to hold the data in cache and shuffle it afterwards. I have not found a wy to shuffle the LMDB databases with the same seed.
        shuffle(allpoints)
 
        # iterates over points of the squares and clips the image acordingly        
        for points in allpoints:
            #clips the label and resizes it to the output dimensions and changes the class house from 255 to one
            label = clip(imInfo1['image'], points, clipSizeLabel, True).astype('int')  
            label = cv2.resize(label, dim, interpolation=cv2.INTER_NEAREST)
            label[label == 255] = 1
            # Prepares the label so it can be read by caffe
            newlabel = np.empty((1,dim[0],dim[1]))
            newlabel[0,:,:] = label
            average = np.average(label)
            #leaves out the imagees where there is less then 10% of the class 1
            if average > 0.1:
                # clips data
                data = clip(imInfo0['image'][:,:,::-1], points, clipSize, False)\
                       .astype('float')
                #prepares the data for the NN dilation     
                img_dat = pre_dilation(data)
                img_dat = caffe.io.array_to_datum(img_dat)
                # writes the data and the labels in two different DB
                with lmdb.open(path_dst + 'Data', map_size=int(1e12)).begin(write=True) as dat_in:
                    dat_in.put('{:0>10d}'.format(index), img_dat.SerializeToString())                
                lab_dat = caffe.io.array_to_datum(newlabel)
                with lmdb.open(path_dst + 'Label', map_size=int(1e12)).begin(write=True) as lab_in:
                    lab_in.put('{:0>10d}'.format(index), lab_dat.SerializeToString())
                index += 1
    print 'finished', path
    return

def pre_dilation(image):
    ''' just transposes the input, to make it accesible for the NN '''
    caffe_in = image.transpose([2, 0, 1])
    return caffe_in

def make_list(rasterpath, labelpath):
    """ creates a list of files that are in the paths rasterpath and labelpath"""
    f = []
    numberOfFiles = 0
    for subdir, dirs, files in os.walk(rasterpath):
        for file in files:
            if file.endswith('.jpg'):
                if os.path.exists(os.path.join(subdir, 'groundtruth', file[:-4] + '.roof')):
                    f.append((os.path.join(subdir, file), (os.path.join(subdir, 'groundtruth', file[:-4] + '.roof'))))
                    numberOfFiles += 1
    print ('files are read into array', numberOfFiles)
    return {'f':f, 'n' :numberOfFiles}

def get_image_info(dataFile):
    """ gets image and finds all kind of information as well as returns a n-dimensional np array
    normalizes imgae between 0 and 1
    Usually uses GDAL to get Inforamtion of the data, but in the case of an error
    it will try to open the file with the help of xdibias. Maybe there is a better way
    but this was the easiest one"""
    try:        
        dsI = gdal.Open(dataFile)

        sizeX = dsI.RasterXSize 
        sizeY = dsI.RasterYSize 
        rCount = dsI.RasterCount
        im = np.float32(dsI.GetRasterBand(1).ReadAsArray())
        #is only needed when the vlaues have to be between 0 and 1
        #im = im / im.max()

        #if image has more than one channel    
        if dsI.RasterCount > 1:
            for (bandnew) in range(2, (rCount+1)):
                new = np.float32(dsI.GetRasterBand(bandnew).ReadAsArray())      
                # is the same aswith im
                #new = new / new.max()
                im = np.dstack((im, new))
     
    except:       
        info  = xdibias.Image(dataFile)
        im = info.readImageData()
        #im = im / im.max()
        sizeX = im.shape[1]
        sizeY = im.shape[0]
        try:
            rCount = im.shape[2]
        except:
            rCount = 1
    return {'xSize':sizeX, 'ySize':sizeY, 'rasterCount':rCount, 'image':im}

def calculates_coord(x, y, degrees, clipSize, stepSize):
    '''calculates squares that are equally distributed within the image. 
       additionally it adds 
        x = image Size x
        y = image Size Y
        degrees = rotation angle
        clipSize = the size of the clips
        stepSize = number of pixels the squares should be separted from each other'''
    points = []
    
    for coordX in range((0+clipSize), (x-clipSize), stepSize):
        for coordY in range(0, y, stepSize):
            for i in range(0, 360, degrees):
            #calculates length of X and Y
                X1 = int(coordX + math.cos(math.radians(i)) * clipSize)
                Y1 = int(coordY - math.sin(math.radians(i)) * clipSize)
                X2 = int(coordX - math.cos(math.radians(i+90)) * clipSize)
                Y2 = int(coordY + math.sin(math.radians(i+90)) * clipSize)
                X3 = int(coordX - math.cos(math.radians(i)) * clipSize)
                Y3 = int(coordY + math.sin(math.radians(i)) * clipSize)
                X4 = int(coordX + math.cos(math.radians(i+90)) * clipSize)
                Y4 = int(coordY - math.sin(math.radians(i+90)) * clipSize)
                singlePointsX = [X1, X2, X3, X4]
                singlePointsY = [Y1, Y2, Y3, Y4]
                
                if all(0 <= item <= x for item in singlePointsX) and \
                all(0 <= item <= y for item in singlePointsY):
                    #more efficent way
                    points.append([[(X1,Y1), (X2,Y2), (X3,Y3), (X4, Y4)],\
                        [45-i],[coordX, coordY]])
                    #points.append([singlePointsX, singlePointsY])                
    print 'finished getting points. Nr. of created squares: ', len(points)  
    return points

def clip(ar, points, clipSize, if_Label):
    '''
    clips the input array to the given points. Because how the images are saved, a black border is created which can be removed by the function crop_around_center
    '''
      
        img = Image.new('F', (ar.shape[1], ar.shape[0]), 0)
        #ImageDraw.Draw(img).polygon([X1, Y1, X2, Y2, X3, Y3, X4, Y4], outline=1, fill=1)
        ImageDraw.Draw(img).polygon(points[0], outline=1, fill=1)        
        mask = np.array(img)
        #create new image
        newImArray = np.empty(ar.shape, dtype=np.float32)        
        #newImArray = ar * mask[:,:, np.newaxis]
        if len(ar.shape) == 3:                   
            newImArray = ar * mask[:,:,np.newaxis]
        else:           
            newImArray = ar * mask[:,:]
        croped_image = crop(newImArray, points[0], ar.shape)
        newImArray = rotate(croped_image, points[1][0],\
                     reshape=False, order=0, mode='constant', cval=0.0)
        if if_Label:
            final = crop_around_center(newImArray, clipSize, points[1][0])   
        else:
            final = crop_around_center(newImArray, clipSize, points[1][0])   
        #counter for fileIndex        
        #save_2_HDF5(inputFile, numberOfFiles, prefix, outputDir, channel, width, height, batch):
        return (final)
   
def crop_around_center(image, clipSize, angle):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """
    width = clipSize
    height = clipSize
    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def crop(image, points, image_size):
    # find biggest and smallest X and Y value and crops the image
    X = []
    Y = []
    for i in points: X.append(i[0])
    for i in points: Y.append(i[1]) 
    maxX = max(X)
    minX = min(X) 
    maxY = max(Y)
    minY = min(Y)
    return image[minY:maxY, minX:maxX]

def save_2_HDF5(dataAr, labelAr, prefix, outputDir, batch):
    '''
    saves the input to HDF5.
    '''
    try:
        hdfFileList=[]  
        widthData = dataAr.shape[0]
        heightData = dataAr.shape[1]
        channelData = dataAr.shape[2]
        
        widthLabel = labelAr.shape[0]
        heightLabel = labelAr.shape[1]
        
        global index  
        global fileIndex

        if (fileIndex % batch) == 0:

            # open and create hdf5 file output directory
            outputHDFFile = prefix + "_" + str(fileIndex) + ".h5"
            #print "file name: " + outputHDFFile
            outputHDFPath = os.path.join(outputDir, outputHDFFile)
            #print "hdf5 file: ", outputHDFPath
            if os.path.exists(outputHDFPath):   
                fileOut = h5py.File(outputHDFPath, 'r+')
            else: 
                fileOut = h5py.File(outputHDFPath, 'w')

            hdfFileList.append(outputHDFPath)

            data = fileOut.create_dataset("data", (batch,channelData, (widthData), heightData), dtype=np.float32)    
            label = fileOut.create_dataset("label", (batch, widthLabel, heightLabel), dtype=np.float32)
            # image data matrix
            #labelStack = np.empty((batch,width,height))
            # initialize index at every batch 
            index=0
        plt.imshow(dataAr)
        #this one here cost me around a mont
        #WICHTIG
        dataAr = dataAr.transpose(2,0,1)

        data[index,:,:,:]=dataAr
        label[index,:,:]=labelAr.astype('float32')

        if (fileIndex % batch) == 0:
            # close the last file
            dataAr.__init__()
            #labelStack.__init__()
            fileOut.close()

            
        outputHDFListFile = prefix + '.txt'
        outputHDFListPath = os.path.join(outputDir, outputHDFListFile)

        if os.path.exists(outputHDFListPath): 
            outputHDFListFile = prefix + '-list.txt'
            outputHDFListPath = os.path.join(outputDir, outputHDFListFile)



        with open(outputHDFListPath, 'a') as trainOut:
            for hdfFile in hdfFileList:
                writeOut=hdfFile + "\n"
                trainOut.write(writeOut)

    except Exception, e:
        print 'Data already exists or '
        print str(e)
        print tb_lineno 
        return
    return(index)

def reset_globals():
    global fileIndex 
    global index    
    fileIndex = 0
    index=0
    return


    


for_HDF5_creation =  (prefix, path_dst, batchSize)

if __name__ == '__main__':
   main(degrees, clipSize, stepSize, rasterPath, labelPath, for_HDF5_creation, path_dst)
