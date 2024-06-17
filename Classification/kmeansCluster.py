import numpy as np
import rasterio as rio
from tslearn import clustering
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from plotGraps import getGraphs
import cv2 as cv
import time
import ipdb

class kMeansCluster:
    def __init__(self,imageFpath , dateFpath, ):
        self.imageFpath = imageFpath
        self.dateFpath = dateFpath
        self.getImage(self.imageFpath)
        self.setDates()
    def getImage(self, imageFpath):
        with rio.open(imageFpath) as dst:
            self.Image = dst.read()
            self.meta = dst.meta
            self.shape = self.Image.shape
        self.flatImg = self.Image.reshape(self.shape[0],-1).T
    def setDates(self,):
        dates = np.loadtxt(self.dateFpath, dtype='str')
        newDates = [datetime.strptime(d,"%Y%m%d").date() for d in dates]
        firstYear = newDates[0].year
        firstDay = datetime(firstYear, 1,1).date()
        julian = [(d-firstDay).days for d in newDates]
        self.dates = newDates
        self.julianDates = julian
    def getElbowCurve(self,clusters=50):
        clusterList = []
        inertiaList = []
        labelList = []
        i = 2
        while i <= n_clusters:
            print(i)
            km_dst = clustering.TimeSeriesKMeans(n_clusters=i, metric="euclidean", 
                n_init=1, n_jobs=5, max_iter=1,max_iter_barycenter=1,).fit(self.flatImg)
            print("K means estimated")
            clusterList.append(i)
            inertiaList.append(km_dst.inertia_)
            labelList.append(km_dst.labels_)
            i+= 1
        plt.plot(clusterList, inertiaList)
        plt.title("Elbow Curve")
        plt.xlabel("Cluster Number")
        plt.ylabel("Inertia")
        plt.show()
        # return clusterList, inertiaList, labelList
    def getKmeansTSL(self, n_clusters=50, ret=True, save=True, savePath=None):
        self.km_tslearn = clustering.TimeSeriesKMeans(
            n_clusters=n_clusters, metric="euclidean", n_init=3, 
            n_jobs=5, max_iter=5,max_iter_barycenter=3,).fit(self.flatImg)
        labelImg = self.km_tslearn.labels_
        labelImg += 1
        self.labelImgTSL = labelImg.reshape(self.shape[1:])
        if ret==True: 
            return self.km_tslearn
        if save is True:
            if savePath is not None:
                meta = self.meta
                meta.update(count=1)
                with rio.open(savePath, 'w', **meta) as dst:
                    dst.write(self.labelImgTSL)
            else:
                print("Save location missing")
    def getKmeansCV(self, n_clusters=50, ret=False, savePath=None):
        #ipdb.set_trace()
        self.n_clusters = n_clusters
        shape = self.shape
        print(self.shape)
        tempImg = np.zeros((shape[1], shape[2], shape[0]))
        for i in range(shape[0]):
            tempImg[:,:,i] = self.Image[i]
        satz = tempImg.reshape((-1, tempImg.shape[2]))
        satz = np.float32(satz)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv.kmeans(satz,n_clusters,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
        label += 1
        labelImg = label.reshape(tempImg.shape[:2])
        self.labelImgCV = labelImg
        if ret==True:
            return labelImg
        if savePath is not None:
            #TODO: Change the data type of input data according to meta file
            meta = self.meta
            meta.update(count=1)
            if savePath==None:
                print("Please enter the save path")
            with rio.open(savePath, 'w', **meta) as dst:
                dst.write(labelImg,1)
    def createGraph(self,saveFolder=None):
        gph = getGraphs(kmobj=self)
        gph.plotAllClusters(saveFolder = saveFolder)

if __name__ == "__main__":
    rasterFile = r"D:\NewImage\Simulation\TestData\stackimg\ndvi_corrected_stack30D.tif"
    dateFile = r"D:\NewImage\Simulation\TestData\dates2.txt"
    saveFile = r"D:\NewImage\Simulation\TestData\clusterOut\cluster_10.tif"
    def testcase0():
        test = kMeansCluster(imageFpath=rasterFile, dateFpath=dateFile)
        start_time = time.time()
        #test.getKmeansCV(n_clusters=10, savePath = saveFile)
        test.getKmeansTSL(n_clusters=10, savePath=saveFile)
        print(f"Time taken in clustering {time.time()-start_time}")
        return test
    test = testcase0()