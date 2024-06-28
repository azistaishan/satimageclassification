import rasterio as rio
from rasterio import mask
import fiona
import pandas as pd
from datetime import datetime
import numpy as np
import copy
import ipdb
import os
from pathlib import Path
class DataFromVectors:
    def __init__(self,shpFilePath=None, rasterFilePath=None, valScale=10000, dateFpath=None):
        if shpFilePath is None:
            self.shpfpath=input(r"Path to Shape File")
        else:
            self.shpfpath=shpFilePath
        if rasterFilePath is None:
            self.imgfpath=input(r"Path to Raster File")
        else:
            self.imgfpath=rasterFilePath
        if dateFpath is None:
            self.TimeSeries = False
            self.dateFpath = None
        else:
            self.TimeSeries = True
            self.dateFpath = dateFpath
        self.openRaster()
        self.openVector()
        self.valScale = valScale
        self.props = []
    def openVector(self,):
        self.shp = fiona.open(self.shpfpath)
        self.shp.keys = self.shp.schema.keys()
        self.shp.no = len(list(self.shp))
    def getKeyDict(self,):
        self.keydict = {}
        k = 0
        for key in self.shp.keys:
            self.keydict[k]=key
            k+=1
    def selectKeys(self, keylist=[]):
        try: 
            len(keylist) != 0
            for i in keylist:
                self.props.append(self.keydict[i])
        except: print('No key selected')

    def openRaster(self,):
        self.dst = rio.open(self.imgfpath)
        self.img = self.dst.read()
        self.imgShape = self.img.shape
    
    def extractData1(self,nofill=-10, key = ['Class'], scaleFactor=1):
        if self.TimeSeries == True:
            try:
                self.julianDates
            except:
                self.getDateToJulian() 
            # one = next(iter(self.shp))
            cols = copy.copy(key)
            valColName = [f"Val_{i}" for i in range(self.imgShape[0])]
            dayColName = [f"Day_{i}" for i in range(len(self.julianDates))]
            cols.extend(valColName)
            cols.extend(dayColName)
            df = pd.DataFrame(columns=cols)
            # print(cols)
            # print(len(cols))
            rowcount = 0
            # In case any shape file is out of bounds, save it in outShp
            self.outShp = []
            for feat in self.shp:
                # tempDf = {}
                shp = [feat['geometry']]
                try:
                    out_img, _ = mask.mask(self.dst, shp, crop=True, filled=True, nodata=nofill)
                    valIdxs = np.where(out_img[0,:,:]!=nofill)
                    for i in range(len(valIdxs[0])):
                        tempList = []
                        arr = out_img[:,valIdxs[0][i],valIdxs[1][i]]*scaleFactor
                        # User can adjust the scaling factor.
                        # print(key)
                        for attr in key:
                            tempList.append(feat['properties'][attr])
                        # tempDf['Vals'] = arr
                        # tempDf['Dates'] = self.julianDates
                        tempList.extend(arr)
                        tempList.extend(self.julianDates)
                        # df.append(tempList)
                        df.loc[rowcount] = tempList
                        # print(tempList)
                        # print(len(tempList))
                        # pdb.set_trace()
                        rowcount += 1
                    #df.append(tempDf)
                    print(f'In bounds {feat}')
                except:
                    print(f'Out of bounds: {feat}')
                    self.outShp.append(feat)
        else:
            cols = copy.copy(key)
            valColName = [f"Val_{i}" for i in range(self.imgShape[0])]
            cols.extend(valColName)
            df = pd.DataFrame(columns=cols)
            rowcount = 0
            self.outShp = []
            for feat in self.shp:
                shp = [feat['geometry']]
                try:
                    out_img, _ = mask.mask(self.dst, shp, crop=True, filled=True, nodata=nofill)
                    valIdxs = np.where(out_img[0,:,:]!=nofill)
                    for i in range(len(valIdxs[0])):
                        tempList = []
                        arr = out_img[:,valIdxs[0][i],valIdxs[1][i]]*scaleFactor
                        for attr in key:
                            tempList.append(feat['properties'][attr])
                        tempList.extend(arr)
                        df.loc[rowcount] = tempList
                        rowcount += 1
                    print(f'In bounds {feat}')
                except:
                    print(f'Out of bounds {feat}')
                    self.outShp.append(feat)
                    pass
        self.rawDf = df

    def extractData(self,folder, nofill=65526, key = ['Class'], scaleFactor=1):
        # ipdb.set_trace()
        if self.TimeSeries == True:
            try:
                self.julianDates
            except:
                self.getDateToJulian() 
            # one = next(iter(self.shp))
            cols = copy.copy(key)
            valColName = [f"Val_{i}" for i in range(self.imgShape[0])]
            dayColName = [f"Day_{i}" for i in range(len(self.julianDates))]
            cols.extend(valColName)
            cols.extend(dayColName)
            tempDict = {}
            for i in cols:
                tempDict[i] = []
            # In case any shape file is out of bounds, save it in outShp
            self.outShp = []
            # pdb.set_trace()
            for feat in self.shp:
                # tempDf = {}
                shp = [feat['geometry']]
                try:
                    out_img, _ = mask.mask(self.dst, shp, crop=True, filled=True, nodata=nofill)
                    # 65526 is standard no data value in Sentinel
                    valIdxs = np.where((out_img[0,:,:]!=nofill))
                    for i in range(len(valIdxs[0])):
                        for j in range(self.imgShape[0]):
                            vect = out_img[:,valIdxs[0][i], valIdxs[1][i]]*scaleFactor
                            tempDict[f'Val_{j}'].append(vect[j])
                        for attr in key:
                            tempDict[attr].append(feat['properties'][attr])
                        for k in range(len(self.julianDates)):
                            tempDict[f'Day_{k}'].append(self.julianDates[k])
                except:
                    print(f'Out of bounds: {feat}')
                    self.outShp.append(feat)
        else:
            cols = copy.copy(key)
            valColName = [f"Val_{i}" for i in range(self.imgShape[0])]
            cols.extend(valColName)
            tempDict = {}
            for i in cols:
                tempDict[i] = []
            self.outShp = []
            count = 0
            for feat in self.shp:
                print(f'The shape used is {feat}')
                shp = [feat['geometry']]
                try:
                    # To make lists of values for pixels in order and create df later
                    out_img, transformed = mask.mask(self.dst, shp, crop=True, filled=True, nodata=nofill)
                    outProfile = self.dst.profile.copy()
                    outProfile.update({
                        'width': out_img.shape[2],
                        'height': out_img.shape[1],
                        'transform': transformed
                    })
                    # folder = r"D:\Ishan\imageProcessing\TestData\testSample"
                    fileName = f"{feat['properties']['Class']}_{count}.tif"
                    totalFile = Path(folder, fileName)
                    with rio.open(totalFile, 'w', **outProfile) as dst:
                        for i in range(out_img.shape[0]):
                            dst.write(out_img[i], i+1)
                    print(f'File {totalFile} should appear now')
                    valIdxs = np.where((out_img[0,:,:]!=nofill))
                    # valIdxs = np.where(out_img[0,:,:]!=nofill)
                    for i in range(len(valIdxs[0])):
                        for j in range(self.imgShape[0]):
                            vect = out_img[:,valIdxs[0][i], valIdxs[1][i]]*scaleFactor
                            tempDict[f'Val_{j}'].append(vect[j])
                        for attr in key:
                            tempDict[attr].append(feat['properties'][attr])
                except:
                    print(f'Out of bounds: {feat}')
                    self.outShp.append(feat)
                count += 1
        self.tempDict = tempDict
        df = pd.DataFrame.from_dict(tempDict)
        self.rawDf = df

    def extractData2(self,nofill=-10):
        if self.TimeSeries is True:
            try:
                self.julianDates
            except:
                self.getDateToJulian() 
            cols = self.props
            cols.append('Vals')
            cols.append('JulianDates')
            df = pd.DataFrame(columns=cols)
            for feat in self.shp:
                tempDf = {}
                shp = [feat['geometry']]
                out_img, _ = mask.mask(self.dst, shp, crop=True, filled=True, nodata=nofill)
                valIdxs = np.where(outImg[0,:,:]!=nofill)
                for i in range(len(valIdxs[0])):
                    arr = out_img[:,valIdxs[0][i],valIdxs[1][i]]
                    for prop in self.props:
                        tempDf[prop] = feature[prop]
                    tempDf['Vals'] = arr
                    tempDf['Dates'] = self.julianDates
                df.append(tempDf)
        self.rawDf = df
    def getDateToJulian(self):
        if 'dates' in self.dst.tags().keys():
            print('Dates found in the dataset')
            dates = np.fromstring(self.dst.tags()['dates'], sep=',')
            newDates = [datetime.strptime(str(d),"%Y%m%d").date() for d in dates]
        elif dateFpath is not None:
            dates = np.loadtxt(self.dateFpath, dtype='str')
            newDates = [datetime.strptime(d,"%Y%m%d").date() for d in dates]
        firstYear = newDates[0].year
        firstDay = datetime(firstYear, 1,1).date()
        julian = [(d-firstDay).days for d in newDates]
        self.dates = newDates
        self.julianDates = julian
    def chooseClass(self,):
        pass
    def saveDataFrame(self, saveFpath = None, format='csv'):
        if format=='csv':
            self.rawDf.to_csv(saveFpath, index=False)
        #TODO
if __name__ == '__main__':
    """
    Testing the scenarios
    """
    shpfpath = r"D:\Ishan\imageProcessing\TestData\shpfile\Testeclasses.shp"
    #"Single image"
    rasterfpath = r"D:\Images\rgbnir_new\20230212T050931_20230212T051514_T44QKF.tif"
    #Imagestack
    #rasterfpath2 = r"D:\Ishan\imageProcessing\TestData\stackimg\ndvi_corrected_stack30D.tif"
    #Date file
    datefpath = r"D:\Ishan\imageProcessing\TestData\dates2.txt"
    # Saving the training data path: 
    csvfpath = r"D:\Ishan\imageProcessing\TestData\trainingSample.csv"
    if os.path.exists(csvfpath):
        print('Deleting old csv file')
        os.remove(csvfpath)
    test = dataFromVectors(shpFilePath=shpfpath, rasterFilePath=rasterfpath)
    # test = dataFromVectors(shpFilePath=shpfpath, rasterFilePath=rasterfpath2, dateFpath=datefpath)
    # test.extractData1()
    test.extractData()
    print('Dataframe created, saving the file...')
    test.saveDataFrame(saveFpath=csvfpath)
    print('File saved')