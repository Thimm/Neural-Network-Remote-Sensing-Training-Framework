import os, math, sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import h5py
import xdibias
import csv
from osgeo import gdal, gdalnumeric, ogr, osr
import osgeo.ogr, osgeo.osr
from random import shuffle
from cv2 import resize, INTER_NEAREST
import subprocess
#import lmdb
#from scipy.misc import imresize
#import caffe
#from utils import get_id_classes, convert_from_color_segmentation, create_lut


#for_HDF5_creation =  (prefix, outputDir, batch, Raster_True)
# def main(degrees, clipSize, stepSize, rasterpath, labelpath, HDF5): 
degrees = 10
clipSize = 900
stepSize = 572

main = '/space/export/data/zwie_th/DLR/Bachelorarbeit/'
rasterPath = '/space/export/data/zwie_th/DLR/TrainingData/Bachelorarbeit_Data/PleiadesData/combined1.tif'
labelPath = main + 'Data/segmentcalc3n.tif'
inputCSV = main + 'Data/OnlyCoord.csv'
save = main + 'Data/Shape/'


locationLabel, locationData = 'Data/label/', 'Data/data/'
outPath = main + 'HDF5_Test/'
prefix = 'dataset'
batchSize = 1
fileIndex = 0
index=0

def get_image_info(dataFile):
    """ gets image and finds all kind of information as well as returns a n-dimensional np array
    normalizes imgae between 0 and 1
    Usually uses GDAL to get Inforamtion of the data, but in the case of an erro
    lsit will try to open the file with the help of xdibias. Maybe there is a better Version
    but this was the easiest one"""
    split_path = dataFile.split('_')
    angle = split_path[-2]    
    try:
        
        dsI = gdal.Open(dataFile)

        sizeX = dsI.RasterXSize 
        sizeY = dsI.RasterYSize 
        rCount = dsI.RasterCount
        im = np.float32(dsI.GetRasterBand(1).ReadAsArray())
    
        im = im / im.max()

        #if image has more than one channel    
        if dsI.RasterCount > 1:
            for (bandnew) in range(2, (rCount+1)):
                new = np.float32(dsI.GetRasterBand(bandnew).ReadAsArray())      
                new = new / new.max()
                im = np.dstack((im, new))

    except Exception, e:
        print str(e)
        print 'which line: ', sys.exc_traceback.tb_lineno
        pass
    return {'xSize':sizeX, 'ySize':sizeY, 'rasterCount':rCount, 'image':im, 'angle':angle}

def make_list(rasterpath, labelpath):
    """ creates a list of files that are in the paths rasterpath and labelpath"""
    f = []
    numberOfFiles = 0
    for subdir, dirs, files in os.walk(rasterpath):
        for file in files:
            if file.endswith('.tif'):
                f.append((os.path.join(subdir, file), (os.path.join(labelpath, file))))
                numberOfFiles += 1
    print ('files are read into array', numberOfFiles)
    return {'f':f, 'n' :numberOfFiles}
    
def create_Polygon_Shp(save, csvFile, r):
    #r = radius of buffer 
    r = math.sqrt(((r/2)**2)*2)
    print r   
    #counter for ID
    counter = 0
    #EPSG Number for right projection
    epsg = 32736
    spatialReference = ogr.osr.SpatialReference() #will create a spatial reference locally to tell the system what the reference will be
    spatialReference.ImportFromEPSG(epsg) #here we define this reference to be utm Zone or Project
    
    #To make a virtual vector container
    driver = osgeo.ogr.GetDriverByName('ESRI Shapefile') # will select the driver foir our shp-file creation.
    print save
    shapeData = driver.CreateDataSource(save) #so there we will store our data
    layer = shapeData.CreateLayer('customs', spatialReference, osgeo.ogr.wkbPolygon)
    layer_defn = layer.GetLayerDefn()
    #not sure why it is here
    #dataset = gdal.Open( "/Users/zwiener/Dropbox/Uni/Geo/Bachelorarbeit/segmap.tif", GA_ReadOnly )
    new_field0 = ogr.FieldDefn('Angle', ogr.OFTInteger)
    new_field1 = ogr.FieldDefn('ID', ogr.OFTInteger)
    layer.CreateField(new_field0)
    layer.CreateField(new_field1)
    
    #opens csv File with coordinates (important: no header!!!)
    file = csvFile
    with open(file) as coordcsv:
        reader = csv.reader(coordcsv, delimiter = ',')
        #goes through all Point and calculates all different edges to clip
        for row in reader:
            coordX = float(row[0])
            coordY = float(row[1])
            for i in range(0, 360, 10):
            #calculates length of X and Y
                X1 = coordX + math.cos(math.radians(i)) * r
                Y1 = coordY - math.sin(math.radians(i)) * r
                X2 = coordX - math.cos(math.radians(i+90)) * r
                Y2 = coordY + math.sin(math.radians(i+90)) * r
                X3 = coordX - math.cos(math.radians(i)) * r
                Y3 = coordY + math.sin(math.radians(i)) * r
                X4 = coordX + math.cos(math.radians(i+90)) * r
                Y4 = coordY - math.sin(math.radians(i+90)) * r
                #Create Ring
                ring = ogr.Geometry(ogr.wkbLinearRing)
                ring.AddPoint(X1, Y1)
                ring.AddPoint(X2, Y2)
                ring.AddPoint(X3, Y3)
                ring.AddPoint(X4, Y4)
        
                #Create Polygon
                poly = osgeo.ogr.Geometry(ogr.wkbPolygon)
                poly.AddGeometry(ring)
                feature = ogr.Feature(layer_defn)
                feature.SetGeometry(poly)
                feature.SetFID(counter)
                layer.CreateFeature(feature)
                feature = layer.GetFeature(counter)
                t = feature.GetFieldIndex('Angle')
                newID = feature.GetFieldIndex('ID')
                feature.SetField(newID, counter)
                feature.SetField(t, i)
                layer.SetFeature(feature)
                del ring, poly
                counter += 1
        
        shapeData.Destroy()
        #Create Spatial Refernce for ESRI shp
        spatialRef = osr.SpatialReference()
        #Seems to work better then EPSG
        spatialRef.ImportFromProj4('+proj=utm +zone=36 +south +datum=WGS84 +units=m +no_defs')
        #more ellegant version could be to just use the wkt file. Instead of writing it to shapefile
        spatialRef.MorphToESRI()
        filex = open('/home/zwie_th/Externe/Bachelorarbeit/Data/Shapefile.prj ', 'w')
        filex.write(spatialRef.ExportToWkt())
        filex.close()
        print 'created Shapefile ', counter
        
    return (save)

def clip(ar, angle, clipSize, if_Label):
    newImArray = rotate(ar, int(angle),\
        reshape=False, order=0, mode='constant', cval=0.0)
    if if_Label:
        final = crop_around_center(newImArray, clipSize, angle)   
    else:
        final = crop_around_center(newImArray, clipSize, angle)   
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
    # find biggest and smallest X and Y value
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
    #initialize
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
        print 'which line: ', sys.exc_traceback.tb_lineno 
        return
    return(index)

def reset_globals():
    global fileIndex 
    global index    
    fileIndex = 0
    index=0
    return

def prep_dilation(data, s, which):
    
    if which:
        # bring to rgb subtract mean of imageNet plus taking the red channel for NIR
        mean_pixel = np.array([102.93, 111.36, 116.52, 116.52], dtype=np.float32)
        data = (data.astype(np.float32) * 255) - mean_pixel
        data[:,:,0:3] = data[:,:,2::-1]
    else:
        dim = (s/8, s/8)
        data = resize(data, dim, interpolation=INTER_NEAREST)
    return data

def clip_via_shp(shp, raster, location, newname, ID, Angle):  
    #gets driver information
    print shp
    try:
        newLocation = location + newname + ID + '_' + Angle + '_' + '.tif'
        call = 'gdalwarp -q -cutline ' + shp + ' -crop_to_cutline' + ' -cwhere FID=' + ID + \
                  ' -of GTiff' + ' -co PROFILE=BASELINE ' + raster + ' ' + newLocation
        print call
        
        subprocess.call(call, shell=True)
    except Exception, e:
       print 'Data already exists or '
       print str(e)
       print sys.exc_traceback.tb_lineno
       return

    return newLocation

def main(save, csvFile, radius, rasterPath, labelPath, locationData, locationLabel, HDF5):  
    #create_Polygon_Shp(save+'shp_data', csvFile, radius)
    #for Dilation
    clipSizeLabel = radius-2*186
    print 'size Data: ', radius
    print 'size Label: ', clipSizeLabel
    #for Dilation
    #create_Polygon_Shp(save+'shp_label', csvFile, clipSizeLabel)
    global fileIndex
    global index
    try:
       driver = ogr.GetDriverByName('ESRI Shapefile')
       dataSource = driver.Open(save+'shp_data', 0) # 0 means read-only. 1 means writeable.
       layer = dataSource.GetLayer()
       if dataSource is None:
          print 'Could not open %s' %(shp)
       #loops through layer, gets angle from attributes and writes file with degree information
       print layer.GetFeatureCount()
       for i in layer:
	   Angle = str(i.GetField('Angle')+90) #because 0 starts at 90 degrees rotation
           ID = str(i.GetField('ID'))
           newLocationD = clip_via_shp(save+'shp_data', rasterPath, locationData, 'dataset', ID, Angle)
           newLocationL = clip_via_shp(save+'shp_label', labelPath, locationLabel, 'dataset', ID, Angle)
           print newLocationD
           imInfo0 = get_image_info(newLocationD)
           imInfo1 = get_image_info(newLocationL)
           
           data = clip(imInfo0['image'], imInfo0['angle'], clipSize, False)
           data = prep_dilation(data, clipSizeLabel, True)
           
           label = clip(imInfo1['image'], imInfo1['angle'], clipSizeLabel, True)     
           label = prep_dilation(label, clipSizeLabel, False)
           save_2_HDF5(data, label, HDF5[0], HDF5[1], HDF5[2])
           fileIndex = fileIndex + 1        
           index += 1
           try:
               
               os.remove(newLocationD)
               os.remove(newLocationL)
               os.remove(newLocationD + '.aux.xml')
               os.remove(newLocationL + '.aux.xml')
           except:
               print 'didnt remove alle the files'
               pass
           print 'finished: ', fileIndex
    except Exception, e:
        print str(e)
        print 'which line: ', sys.exc_traceback.tb_lineno
        fileIndex = fileIndex + 1        
        index += 1
        pass
   #     #plt.figure(0)
   #     #plt.imshow(data)
   #     #plt.figure(1)
   #     #plt.imshow(label)
   #     #plt.show()          
          
    return

for_HDF5_creation =  (prefix, outPath, batchSize)
# def main(degrees, clipSize, stepSize, rasterpath, labelpath, HDF5):  
main(save, inputCSV, clipSize, rasterPath, labelPath, locationData, locationLabel, for_HDF5_creation)
