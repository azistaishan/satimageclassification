import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import matplotlib.dates as mdates

class getGraphs:
    """
    Plot graphs for kmeans clustering 

    Attributes:
        kmobj: Object setup using kmeansCluster.py
        loadFiles: If kmeans object is not setup, then load files
        dateFpath: If kmeans object is not setup, then load date files
        imgFpath: If kmeans object is not setup, then load the image File path
        clusterImgFpath: Image of cluster image

    Methods:
        loadFiles
        plotClusterId
        plotAllClusters
    """
    def __init__(self,kmobj=None, loadFiles=False, dateFpath = None, imgFpath = None, clusterImgFpath = None):
        """
        Constructs all the necessary attributes for the object

        Parameters:
        
        kmobj: Object setup using kmeansCluster.py
        loadFiles: If kmeans object is not setup, then load files
        dateFpath: If kmeans object is not setup, then load date files
        imgFpath: If kmeans object is not setup, then load the image File path
        clusterImgFpath: Image of cluster image

        Returns:
        None
        """
        if kmobj is not None:
            self.clusterImg = kmobj.labelImgTSL
            self.dates = kmobj.dates
            self.img = kmobj.Image
        else:
            self.loadFiles(dateFpath=dateFpath, imgFpath = imgFpath, clusterImgFpath=clusterImgFpath)
    def loadFiles(self, dateFpath=None, imgFpath=None, clusterImgFpath=None):
        """
        Loads files to the object:

        Parameters:
        
        dateFpath: If kmeans object is not setup, then load date files
        imgFpath: If kmeans object is not setup, then load the image File path
        clusterImgFpath: Image of cluster image

        Retruns:
        None
        """
        if dateFpath is None:
            self.datesFpath = input(r"Date file to plot trend: ")
        else:
            self.datesFpath = dateFpath
        if imgFpath is None:
            self.imageStackFpath = input(r"Image stack to plot the trend: ")
        else:
            self.imageStackFpath = imgFpath
        if clusterImgFpath is None:
            self.clusterFpath = input(r"Cluster image to select pixels: ")
        else: self.clusterFpath = clusterImgFpath
        with rio.open(self.clusterFpath) as dst:
            self.clusterImg = dst.read(1)
            self.cMeta = dst.meta
        tempDates = np.loadtxt(dateFpath, dtype='str')
        self.dates = [datetime.strptime(d, '%Y%m%d').date() for d in tempDates]
        with rio.open(self.imageStackFpath) as dst:
            self.img = dst.read()
            self.imgMeta = dst.meta
    def plotClusterId(self, clusterID, random=1000, ymin=-0.1, ymax=0.7, saveFpath=None, showPlot=False):
        """
        Plots the trend for clusterID selected with number of plots on random set by random

        Parameters:

            clusterID: Int
                Cluster number. If 10 clusters are there, then number between 1-10. 0 is avoided
            random: Int
                Number of pixels to select in random to be plotted
            ymin: Float
                Minimum of the y axis to plot
            ymax: Float
                Maximum of the y axis to plot
            saveFpath: Str
                Path to save the plot
            showPlot: Bool
                Option to show plot, to be set to False if save is selected 
        Returns:
            Plot if showPlot is True
        """
        temp = self.img[:,self.clusterImg==clusterID]
        randomIdxs = np.random.choice(range(temp.shape[1]), size = random)
        plt.clf()
        plt.ioff()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
        for i in randomIdxs:
            plt.plot(self.dates, temp[:,i])
        plt.ylim(ymin, ymax)
        plt.xticks(self.dates, rotation=70)
        plt.grid(axis='x')
        if saveFpath is not None:
            plt.savefig(saveFpath)
        if showPlot is True:
            plt.show()
        plt.clf()
    def plotAllClusters(self, saveFolder, random=1000, ymin=-0.1, ymax=0.7):
        """
        Plot all clusters defined in the cluster file

        Parameters: 
            saveFolder: Str
                Path to the folder where the plots should be saved
            random: Int
                Number of pixels to select in random to be plotted
            ymin: Float
                Minimum of the y axis to plot
            ymax: Float
                Maximum of the y axis to plot
        Returns:
            None
        """
        for i in np.unique(self.clusterImg):
            saveFpath = Path(saveFolder, str(i)+'.jpg')
            self.plotClusterId(clusterID=i, random=random, ymin=ymin, ymax=ymax, saveFpath=saveFpath)
