import numpy as np
import rasterio as rio
from pathlib import Path
from tslearn import clustering
import matplotlib.pyplot as plt
import pickle
import math
import random
import copy
from datetime import datetime
import matplotlib.dates as mdates
import ipdb
import pandas as pd
class imagesimulation:
    """
    In this class,
    Docstring work in progress
    TODO
    """
    def __init__(self, refImg = None, paramsFile = None):
        if refImg is not None:
            with rio.open(refImg) as dst:
                self.Img = dst.read(1)
                self.meta = dst.meta
                self.imgProfile = dst.profile
        if paramsFile is not None:
            self.loadParams(fileName=paramsFile)
        else:
            print('No parameter files given')
    def gaussian(self,x, amp, mu, sig):
        return (
            amp/ (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
        )
    def invgaussian(self,x, amp, mu, sig):
        return -1*self.gaussian(x, amp, mu, sig)
    def linear(self,x, theta, c):
        # print('Theta is:\t',theta)
        m = math.tan(math.radians(theta))
        return(m*x + c)
    def sigmoid(self,x, ctr, wf ):
        """
        wf: Width factor
        ctr: Center
        """
    
        return (1+np.exp(-(x-ctr)/wf))**-1
    def invsigmoid(self,x, ctr, wf):
        return 1 - self.sigmoid(x,ctr,wf)
    def phenocurve3(self,x, **kwargs):
        # print('Here in phenocurve3, kwargs: \n', kwargs)
        # print(kwargs['theta'])
        line = self.linear(x, theta=kwargs['theta'], c = kwargs['c'])
        gauss1 = self.invgaussian(x, amp=kwargs['amp1'],mu=kwargs['mu1'], sig=kwargs['sigma1'])
        gauss2 = self.gaussian(x, amp=kwargs['amp2'], mu = kwargs['mu2'], sig = kwargs['sigma2'])
        sigmoid1 = self.invsigmoid(x, ctr= kwargs['mu1'] - kwargs['offmu1'], wf=kwargs['wf1'])
        sigmoid2 = self.invsigmoid(x, ctr= kwargs['mu2'] - kwargs['offmu2'], wf=kwargs['wf2'])
        y = line + gauss1*sigmoid1+gauss2*sigmoid2
        return y
    def plotwithRandom(self, sigma, numOfPoints=None, n=100, 
                        save=False, fileName = None, showPlot=False,y_lim = (-0.1,1),
                        dateFile=None, **kwargs):
        """
        If datefile is given then num of Points is not considered.
        """
        plt.clf()
        plt.ioff()
        if dateFile is not None:
            dates = np.loadtxt(dateFile, dtype='str')
            dates = [datetime.strptime(d,'%Y%m%d').date() for d in dates]
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
            numOfPoints = len(dates)
            x = np.arange(numOfPoints) 
        else:
            x = np.arange(numOfPoints)
            dates = x
        #sigmaArr = list(map(lambda x: x + sigma*np.random.rand(), np.zeros(len(x))))
        sigmaArr = np.zeros(numOfPoints)+sigma
        # print(sigmaArr)
        # print('Here at 73, kwargs: \n', kwargs)
        curve =self.phenocurve3(x,**kwargs) 
        i = 0
        # ipdb.set_trace()
        while i < n:
            random.seed(np.random.randint(100))
            #print(curve)
            # print(sigmaArr)
            out = list(map(random.gauss,curve, sigmaArr))
            plt.plot(dates, out)
            i+=1
        plt.ylim(y_lim[0],y_lim[1])
        plt.xticks(dates, rotation=70)
        if save is True:
            plt.savefig(fname=fileName, bbox_inches='tight')
        if showPlot is True:
            plt.show()
    def plotAllParams(self, sigma=0.05, num_of_rand=1000, dateFile=None, save=True, 
    folder=None):
        """
        Parameter file needed 
        """
        plotList = []
        for i in self.paramsDict.keys():
            fileName = f'Plot_{i}.png'
            filePath= Path(folder,fileName)
            self.plotwithRandom(sigma=sigma, 
            n=num_of_rand, save=True,
            fileName=filePath,
            dateFile=dateFile, **self.paramsDict[i])
            print('Images saved for key, ',i)

    def genwithRandom(self,x, sigma, n=100, **kwargs):
        #sigmaArr = list(map(lambda x: x + sigma*np.random.rand(), np.zeros(len(x))))
        sigmaArr = np.zeros(len(x))+sigma
        # print(sigmaArr)
        curve =self.phenocurve3(x,**kwargs) 
        #plt.plot(x, curve)
        i = 0
        tofill = np.zeros((n,len(x)))
        while i < n:
            random.seed(np.random.randint(100))
            out = list(map(random.gauss,curve, sigmaArr))
            tofill[i] = out
            i+=1
        return tofill
    def loadParams(self,fileName):
        with open(fileName, 'rb') as handle:
            self.paramsDict = pickle.load(handle)
    def fillClusters(self,imgnum=30, sigma=0.05):
        """
        clusterFile: File path to the image with clusters defined
        paramsDict: Dictionary of parameters used to generate phenotype
        imgnum: Number of images to be stacked
        sigma: Standard Deviation to be applied for 
        """
        x = np.arange(imgnum)
        # with rio.open(clusterFile) as dst:
        #     image = dst.read(1)
        #     meta = dst.meta
        #     profile = dst.profile
        print('Sigma for filling is: ',sigma )
        inpimg = np.zeros((imgnum, self.Img.shape[0], self.Img.shape[1]))
        for i in self.paramsDict.keys():
            print('Filling cluster: ', i)
            params = self.paramsDict[i]
            idxs = np.where(self.Img==i)
            idx_num = len(idxs[0])
            tofill = self.genwithRandom(x, sigma, n=idx_num, **params)
            for j in range(idx_num):
                inpimg[:,idxs[0][j],idxs[1][j]] = tofill[j]
        return inpimg
    def saveclusters(self,outfile, imgnum=30, sigma=0.05):
        # print('Saving the fille for sigma: ',sigma)
        check = self.fillClusters(imgnum=imgnum, sigma=sigma)
        profile = copy.copy(self.imgProfile)
        profile.update(count=imgnum)
        with rio.open(outfile, 'w', **profile) as dst:
            dst.write(check)
        print('File written')

class simulationTraining:
    def __init__(self,classImgFile, simImgFile, trainingperclass=10000):
        with rio.open(classImgFile) as dst:
            self.classImg = dst.read(1)
        with rio.open(simImgFile) as dst:
            self.simImg = dst.read()
        print('Size of classsImage: ', self.classImg.shape)
        print('Size of simImg: ', self.simImg.shape)
        print('Checking for the classes...')
        self.classes = np.unique(self.classImg)
        print('Classes in classImg',self.classes)
        self.trainingperclass = trainingperclass
    def getTraining(self):
        #Setting the dictionary
        d = {}
        d['Class'] = []
        for i in range(len(self.simImg)):
            d[f'Val_{i}'] = []
        for c in self.classes:
            print(f'At class {c}')
            idxs = np.where(self.classImg==c)
            iterLen = min(self.trainingperclass, len(idxs[0]))
            randId = np.random.choice(range(len(idxs[0])), size=iterLen)
            # print(f'randId len: {len(randId)}')
            # print(f'randID: \n', randId)
            for r in randId:
                arr = self.simImg[:,idxs[0][r], idxs[1][r]]
                d['Class'].append(c)
                for i in range(len(self.simImg)):
                    d[f'Val_{i}'].append(arr[i])
        return pd.DataFrame.from_dict(d)   


if __name__ == "__main__":
    tenclassImg = r"D:\Ishan\imageProcessing\TestData\clusterOut\tenclassimg.tif"
    params = r"D:\NewImage\Simulation\ParameterCurves\parameterDict.pkl" 
    dateFile = r"D:\Ishan\imageProcessing\TestData\dates2.txt"
    def testcase1():
        #tenclassImg = r"D:\Ishan\imageProcessing\TestData\clusterOut\tenclassimg.tif"
        #params = r"D:\NewImage\Simulation\ParameterCurves\parameterDict.pkl"
        test = imagesimulation(refImg=tenclassImg, paramsFile=params)
        sigma = 0.25
        outfile = r"D:\NewImage\Simulation\TestImages\simulatedImg_sigma-dot25.tif"
        test.saveclusters(outfile=outfile, sigma=sigma)
    def testcase2():
        """
        Plot graphs
        """
        test = imagesimulation(refImg=tenclassImg, paramsFile=params)
        folder = r"D:\NewImage\Simulation\SimulationCurves\dot18"
        sigma = 0.18
        print('Value of sigma: ', sigma)
        test.plotAllParams(sigma=sigma, num_of_rand=1000,dateFile=dateFile, save=True,
        folder=folder
        )
    def testcase3():
        tenclassImgFile = r"D:\Ishan\imageProcessing\TestData\clusterOut\tenclassimg.tif"
        simImgFile = r"D:\NewImage\Simulation\TestImages\simulatedImg_sigma-dot25.tif"
        test = simulationTraining(classImgFile=tenclassImgFile, simImgFile=simImgFile)
        out = test.getTraining()
        outFile = r"D:\NewImage\Simulation\TrainingData\sigma-dot25.csv"
        out.to_csv(path_or_buf=outFile, index=False)
    #testcase1()
    testcase2()