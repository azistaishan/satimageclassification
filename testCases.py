from randomForest import RandomForest
from supportVectorMach import SVMClassification
from neuralNet import NeuralNet
from maxlikelihood import MLClassifier
from kmeansCluster import kMeansCluster
import testConfig as tc
import time
from datafromvectors import dataFromVectors
import ipdb, os, shutil
def testcase0():
    """
    Create test data from Image and vecro overlaid
    """
    pass

def rfcTestCase1():
    """
    Model builder for Random Forest using a single day data
    Classify image using the fitted model
    Save model after fitting using the training dataset,

    1 Day multi-band model
    """
    test = RandomForest(
        dfFpath=tc.trainingcsv1D,
        extract=True,
        )
    start_time = time.time()
    test.fit(params=tc.rfcParams)
    print(f"Time taken for model fitting: {time.time()-start_time}")
    classify_time = time.time()
    test.classifyimg(outFpath=tc.rfcOutRaster1D, imgFpath=tc.inRaster1D)
    print(f"Time taken to classify image: {time.time()-classify_time}")
    test.saveModel(saveFpath=tc.rfcModelOutPath1D)
    test.classification_model()
def rfcTestCase2():
    """
    Model builder for Random Forest using 30 day data
    Classify image using the fitted model
    Save model after fitting using the training dataset
    30 Day time series model
    """
    test = RandomForest(
        dfFpath=tc.trainingcsv30D,
        extract = True,
    )
    start_time = time.time()
    test.fit(params=tc.rfcParams)
    print(f"Time taken for model fitting: {time.time()-start_time}")
    classify_time = time.time()
    test.classifyimg(outFpath=tc.rfcOutRaster30D)
    test.saveModel(saveFpath=tc.rfcModelOutPath30D)

def rfcTestCase3():
    """
    In Progress
    """
    pass

def nnTestCase1():
    """
    Model builder for Neural Network using 1 Day data
    Classify image using the fitted model
    Save Model after fitting using the training dataset
    1 Day MultiSpectral model
    """
    test = NeuralNet(dfFpath=tc.trainingcsv1D, extract=True)
    start_time = time.time()
    test.fit()
    print(f"Time taken for model fitting: {time.time()-start_time} secs")
    test.classification_model()
    test.classifyImg(imgFpath=tc.inRaster1D, outFpath=tc.nnOutRaster1D)

def nnTestCase2():
    """
    Model builder for Neural Network using 30 Days data
    Classify image using the fitted model
    Save model after fitting using the training dataset
    30 Day Time Series Model
    """
    test = NeuralNet(dfFpath=tc.trainingcsv30D, extract=True)
    start_time = time.time()
    test.fit()
    print(f"Time taken for model fitting: {time.time()-start_time} secs")
    test.classification_model()
    test.classifyImg(imgFpath=tc.inRaster30D, outFpath=tc.nnOutRaster30D)

def nnTestCase3():
    """
    In Progress
    #TODO
    """
    pass
def dfvTest0():
    """
    Flexible file input to give data from vectors
    """
    # ipdb.set_trace()
    shapeFile = "/home/ishan/workingDir/AFR/classes/classes.shp"
    rasterFile = "/home/ishan/workingDir/AFR/ClippedRaster/1r121_5b_composite_clipped.tif"
    test = dataFromVectors(shpFilePath=shapeFile, rasterFilePath=rasterFile)
    csvFile = "/home/ishan/workingDir/AFR/TrainingData/trainingdata.csv"
    folderPath = "/home/ishan/workingDir/AFR/TrainingData/tempFolder"
    print('Extracting Data')
    test.extractData(folder=folderPath)
    print('Saving data frame')
    test.saveDataFrame(saveFpath=csvFile)
def dfvTest1():
    """
    Flexible file input to give data from vectors
    """
    # ipdb.set_trace()
    # shapeFile = input(r'Path to the shape file: ')
    # rasterFile = input(r'Path to the raster file: ')
    shapeFile = "/home/ishan/workingDir/AFR2/Data/AOI.shp"
    rasterFile = "/home/ishan/workingDir/AFR2/Data/2r170_5b_composite_clip.tif"
    # test = dataFromVectors(shpFilePath=shapeFile, rasterFilePath=rasterFile)
    # folderPath = input('Folder for temporary files: ')
    folderPath = "/home/ishan/workingDir/AFR2/Temp"
    csvFile = f"{folderPath}/trainingdata.csv"
    print('Extracting Data')
    # test.extractData(folder=folderPath)
    print('Saving data frame')
    # test.saveDataFrame(saveFpath=csvFile)
    out = RandomForest(dfFpath=csvFile, extract=True, underSample=True)
    start_time = time.time()
    out.fit(params=tc.rfcParams)
    print(f"Time taken for model fitting: {time.time()-start_time} secs")
    print("---\n---\n---\n Saving Image\n---\n---\n---")
    out.classifyimg(imgFpath=rasterFile, outFpath=tc.rfcOutRaster1D)
    print("!!!- Image Saved -!!!")
    out.classification_model()
    return out
def dfvTest2():
    """
    Flexible file input to give data from vectors
    """
    # ipdb.set_trace()
    # shapeFile = input(r'Path to the shape file: ')
    # rasterFile = input(r'Path to the raster file: ')
    folderPath = "/home/ishan/workingDir/AFR2/Temp"
    csvFile = f"{folderPath}/trainingdata.csv"
    shapeFile = "/home/ishan/workingDir/AFR2/Data/AOI2/AOI_2.shp"
    rasterFile = "/home/ishan/workingDir/AFR2/Data/2r170_5b_composite_clip.tif"
    # test = dataFromVectors(shpFilePath=shapeFile, rasterFilePath=rasterFile)
    # print('Removing temp files from previous run')
    # shutil.rmtree(folderPath)
    # os.makedirs(folderPath)
    # print('Files removed... ')
    # print('Extracting Data')
    # test.extractData(folder=folderPath)
    # print('Saving data frame')
    # test.saveDataFrame(saveFpath=csvFile)
    out = NeuralNet(dfFpath=csvFile, extract=True, underSample=True)
    start_time = time.time()
    out.fit()
    print(f"Time taken for model fitting: {time.time()-start_time} secs")
    out.classification_model()
    out.classifyImg(imgFpath=rasterFile, outFpath=tc.nnOutRaster1D)