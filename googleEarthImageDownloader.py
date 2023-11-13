from datetime import datetime
import ee
import geemap
import os

class GeeImageDownloader:
    def __init__(self,):
        try: 
            ee.Initialize()
        except: 
            ee.Authenticate()
            ee.Initialize()
    def setROI(self,):
        roiFpath = input(r"Input the polygon asset GEE path: ")
        stDate = input(r"Start Date (YYYYMMDD): ")
        enDate = input(r"End Date (YYYYMMDD): ")
        self.roi = ee.FeatureCollection(roiFpath)
        stDate = datetime.strptime(stDate, "%Y%m%d")
        enDate = datetime.strptime(enDate, "%Y%m%d")
        self.start = ee.Date.fromYMD(stDate.year, stDate.month, stDate.day)
        self.end = ee.Date.fromYMD(enDate.year, enDate.month, enDate.day)

    def getImagesCollection(self, cloudPercent = 1):
        self.images = ee.ImageCollection('COPERNICUS/S2') \
            .filterBounds(roi) \
            .filterDate(self.start, self.end) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudPercent))
    def downloadNDVI(self,out_dir):
        ndvi = images.map(lambda image: image.normalizedDifference(['B8', 'B4']))
        geometry = self.roi.geometry()
        geemap.download_ee_image_collection(ndvi, out_dir, scale=10,region=self.roi)
    def downloadRNDVI(self, out_dir):
        rndvi = images.map(lambda image: image.normalizedDifference(['B8', 'B5']))
        geometry = self.roi.geometry()
        geemap.download_ee_image_collection(ndvi, out_dir, scale=10,region=self.roi) 

if __name__ == "__main__":
    x = GeeImageDownloader()