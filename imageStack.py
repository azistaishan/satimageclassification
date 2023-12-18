import rasterio
import matplotlib.pyplot as plt
import pathlib
import geopandas as gpd
import fiona
from pathlib import Path
import numpy as np
import os
import glob
class ImageStack:
    def __init__(self):
        pass

    def image_stack(self):
        """
        To stack rgb and nir image
        """
        rgbfpath = input(r"Path to rgb image")
        nirfpath = input(r"Path to nir file")
        outputfolder = input(r"Path to output folder")

        rgbimg = rasterio.open(rgbfpath)
        nirimg = rasterio.open(nirfpath)
        #check if the file names are same and extract the file name
        rgbfname = Path(rgbfpath).name
        nirfname = Path(nirfpath).name
        try: rgbfname == nirfname
        except NameError:
            print("Filenames do not match")
        outfname = rgbfname
        outfpath = Path(outputfolder, outfname)
        rgbarray = rgbimg.read()
        nirarray = nirimg.read()
        stackedarray = np.dstack((rgbarray, nirarray))
        meta = rgbimg.meta
        with rasterio.open(outfpath, 'w', **meta) as dst:
            dst.write(stackedarray)

    def rgbnirFolderstack(self):
        rgbfolder = input(r"rgb image folder: ")
        nirfolder = input(r"nir image folder: ")
        outputfolder = input(r"output folder: ")
        rgbfiles = os.listdir(rgbfolder)
        filesAbsent = []
        for i in rgbfiles:
            if os.path.isfile(Path(nirfolder,i))==False:
                print(i)
                filesAbsent.append(i)
        try:
            len(filesAbsent) == 0
        except OSError:
            print("Following files are absent in nir folder")
            print(filesAbsent)
        for i in rgbfiles:
            rgbfPath = Path(rgbfolder, i)
            nirfPath = Path(nirfolder, i) 
            outfPath = Path(outputfolder, i)
            rgbimg = rasterio.open(rgbfPath)
            nirimg = rasterio.open(nirfPath)
            b = rgbimg.read(1)
            # print(b.shape)
            g = rgbimg.read(2)
            # print(g.shape)
            r = rgbimg.read(3)
            # print(r.shape)
            nir = nirimg.read(1)
            # print(nir.shape)
            # stackedarray = np.dstack((r,g,b,nir))
            meta = rgbimg.meta
            meta.update({
                'count': 4
            })
            with rasterio.open(outfPath, 'w', **meta) as dst:
                dst.write(r, 1)
                dst.write(g, 2)
                dst.write(g, 3)
                dst.write(nir, 4)

    def FCCFolderstack(self):
        rgbfolder = input(r"rgb image folder: ")
        nirfolder = input(r"nir image folder: ")
        outputfolder = input(r"output folder: ")
        # rgbfiles = glob.glob(rgbfolder+"/*.tif")
        rgbfiles = os.listdir(rgbfolder)
        filesAbsent = []
        for i in rgbfiles:
            if os.path.isfile(Path(nirfolder,i))==False:
                print(i)
                filesAbsent.append(i)
        try:
            len(filesAbsent) == 0
        except OSError:
            print("Following files are absent in nir folder")
            print(filesAbsent)
        for i in rgbfiles:
            rgbfPath = Path(rgbfolder, i)
            nirfPath = Path(nirfolder, i) 
            outfPath = Path(outputfolder, i)
            rgbimg = rasterio.open(rgbfPath)
            nirimg = rasterio.open(nirfPath)
            b = rgbimg.read(1)
            # print(b.shape)
            g = rgbimg.read(2)
            # print(g.shape)
            r = rgbimg.read(3)
            # print(r.shape)
            nir = nirimg.read(1)
            # print(nir.shape)
            # stackedarray = np.dstack((r,g,b,nir))
            meta = rgbimg.meta
            with rasterio.open(outfPath, 'w', **meta) as dst:
                dst.write(nir, 1)
                dst.write(r, 2)
                dst.write(b, 3)
            print(i)

    def cloudMaskTest(self):
        rgbFpath = input(r"Input rgbFpath Name")
        outCloudFolder = input(r"input could mask folder: ")
        minThresG = 3000
        rgbimg = rasterio.open(rgbFpath)
        g = rgbimg.read(2)
        meta = rgbimg.meta
        meta.update({
            'count': 1
        })
        while minThresG < 4000:
            maskarray = np.zeros(g.shape)
            maskarray[g>minThresG] = 1
            Fname = Path(rgbFpath).stem
            ext = Path(rgbFpath).suffix
            newFname = Fname+f"_{minThresG}"+ext
            newFpath = Path(outCloudFolder,newFname)
            with rasterio.open(newFpath, 'w', **meta) as dst:
                dst.write(maskarray,2)
            minThresG += 50
    def getcloudMask(self, imgPath, outFolder, thres=3000, band='Green'):
        """
        Creates the cloud masked image using RGBNir Image 
        input:
            Image to be cloud corrected
        Output:
            Cloud Mask with same name as the RGB Nir Image with _Cloud.
        """
        with rio.open(imgPath) as dst:
            img = dst.read()
            meta = dst.meta
            imgShape = img.shape
        temp = np.zeros(imgShape[1], imgShape[2])
        temp[img[2,:,:]>=thres] = 1
        givenP = Path(imgPath)
        newName = f"{givenP.stem}_Cloud{givenP.suffix}"
        outFpath = Path(outFolder, newName)
        meta.update(count=1)
        with rio.open(outFpath, 'w', **meta) as dst:
            dst.write(temp,1)
    def applyCloudMask(self, imgPath, cloudMaskPath, outFolder, fillVal = 0):
        with rio.open(imgPath) as dst:
            img = dst.read()
            meta = dst.meta
            imgShape = img.shape
        with rio.open(cloudMaskPath) as dst:
            cloudM = dst.read()
            #cloudMeta = dst.meta
            cloudShape = dst.shape
        try:
            cloudShape[1] == imgShape[1]
            img[:,cloudM==1] = fillVal
            givenN = Path(outFolder)
            Fname = Path(imgPath).stem
            ext = Path(imgPath).suffix
            outFname = f"{Fname}_CloudFixed{ext}"
            outFpath = Path(outFolder,outFname)
            with rio.open(outFpath, 'w', **meta) as dst:
                dst.write(img)
        except:
            Print('The shape of cloud and image do not match')

    def getImageStack(self, imgFolder, outFolder, outFname, ext='tif'):
        """
        This stack applies for single value maps such as NDVI, NDRE image stack
        """
        imageList = glob.glob(f"imgFolder\*.{ext}")
        imgSize = len(imageList)
        with rio.open(imageList[0]) as dst:
            img = dst.read()
            imgShape = img.shape
            meta = dst.meta
        zeroImg = np.zeros(imgSize, imgShape[1], imgShape[2])
        for i in range(len(imageList)):
            with rio.open(imageList[i]) as dst:
                zeroImg[i] = dst.read(1)
        meta.update(count = imgSize)
        newImgFpath = Path(outFolder, outFname)
        with rio.open(newImgFpath, 'w', **meta) as dst:
            dst.write(zeroImg)
    def maskImageStack(self, imgFpath, maskFpath, outFpath, maskVal = 1, maskFill = 0):
        with dst.open(imgFpath) as dst:
            img = dst.read()
            meta = dst.meta
            imgShape = img.shape
        with dst.open(maskFpath) as dst:
            mask = dst.read(1)
        img[:,mask==maskVal] = maskFill
        with dst.open(outFpath, 'w', **meta) as dst:
            dst.write(img)
