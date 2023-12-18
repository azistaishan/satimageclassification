from randomForest import RandomForest
from supportVectorMach import SVMClassification
from neuralNet import NeuralNet
from maxlikelihood import MLClassifier
from kmeansCluster import kMeansCluster
import testConfig as tc
import time


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
        dfFpath=tc.trainingcsv1Dm,
        extract=True,
        )
    start_time = time.time()
    test.fit(params=tc.rfcParams)
    print(f"Time taken for model fitting: {time.time()-start_time}")
    classify_time = time.time()
    test.classifyimg(outFpath=tc.rfcOutRaster1D)
    print(f"Time taken to classify image: {time.time()-classify_time}")
    test.saveModel(saveFpath=tc.rfcModelOutPath1D)

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
    """
    pass