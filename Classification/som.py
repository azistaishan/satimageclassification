import rasterio as rio
import time
import numpy as np
from minisom import MiniSom
from datetime import datetime
from plotGraps import getGraphs
import copy
class SOMClassifier
"""
#TODO
"""
def __init__(self,imageFpath, dateFpath):
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
def setSom(self,somShape=(30,30), learningRate=0.5, iterations=100):
    self.som = MiniSom(somShape[0], self.shape[0], learning_rate=learningRate)
    self.som.train(self.flatArr, num_iteration=iterations, verbose=True)
    winner_coordinates = np.array([self.som.winner(x) for x in flat]).T
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
    # Changing the cluster indices to values starting from 1
    