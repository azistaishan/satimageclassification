import rasterio as rio
from rasterio import mask
from rasterio.merge import merge
import numpy as np
import fiona
import ImageOperations.BandProfile as bf

def openVector(shp_file):
    shp = fiona.open(shp_file)
    shp.keys = shp.schema.keys()
    shp.no = len(list(shp))
    return shp

def openRaster(img_fpath):
    dst = rio.open(img_fpath)
    img = dst.read()
    meta = dst.meta
    profile = dst.profile
    return img, meta, profile, dst

def clipFromVector(raster_file, shp_file, out_file, **kwargs):
    """
    Clip a raster from vector. 
    Current options limited
    Saves to a file

    Parameters:
    raster_file: str
        Path to the raster file to be clipped
    shp_file: str
        Path to the shape file to be used to clip raster
    out_file: str
        Path to the output file 
    **kwargs: dictionary
        Keywords: 
            'no_fill_value': int/None/NAN 
            'crop': boolean
            'filled': Boolean

    """
    with rio.open(raster_file) as dst:
        image = dst.read()
        meta = dst.meta()
    shp = openVector(shp_file)
    dst = rio.open(raster_file) 
    try: nofill = kwargs['no_fill']
    except: 
         print('Default No fill value: 0')
         nofill = 0
    out_file_list = []
    for feat in shp:
        vector = [feat['geometry']]
        out_img, transformed = mask.mask(dst, vector, crop=True, filled=True, nodata=nofill)
        out_profile = dst.profile.copy()
        out_profile.update({
             'width': out_img.shape[2],
             'height': out_img.shape[1],
             'transform': transformed
        })
        dst_out = rio.open(out_file, 'w', **out_profile)
        for i in range(out_img.shape[0]):
             dst_out.write(out_img[i],i+1)
        dst.close()
        out_file_list.append(out_file)
    return out_file_list

def mosaicRaster(files_list, out_file, **kwargs):
    mosaic, mosaic_transform = merge(files_list)
    with rio.open(files_list[0]) as dst:
        meta = dst.meta()
    out_meta = meta.copy()
    out_meta.update({'height':mosaic.shape[1],
                     'width':mosaic.shape[2],
                     'transform':mosaic_transform})
    dst = rio.open(out_file, 'w', **out_meta)
    dst.write(mosaic)
    dst.close()
    return out_file 

def stackImages(files_list, out_file, **kwargs):
    #Check the num of files, atleaset >=2
    if len(files_list)<2:
        return 'Only single file in the list'
    stacked,_,profile,_ = openRaster(files_list[0])
    for i in files_list[1:]:
        img,_,_,_ = openRaster(i)
        stacked = np.dstack((stacked, img))
    profile.update({'count':stacked.shape[0]})
    dst = rio.open(out_file, 'w', **profile)
    dst.write(stacked)
    dst.close()

def calculate_func(image_file,
                   out_file,
                   raster_func_list, 
                   add_new_layer=True,
                   band_dict =bf.sent_bands_dict):
    image,_,profile,_ = openRaster(image_file)
    band_profile = bp.update_band_profile(band_dict, image)
    if len(raster_func_list) >1 and add_new_layer is True:
        raise ValueError('add_new_layer cannot be False when raster_func_list has more than 1 function')
    for raster_func in raster_func_list:
        calc_img = eval(raster_func, band_profile)
    if add_new_layer is True:
        out_img = np.dstack(image, calc_img)
        profile.update({'count': profile['count']+1})
    else:
        out_img = calc_img
        profile.update({'count': 1})
    dst = rio.open(out_file, 'w', **profile)
    dst.write(out_img)
    dst.close()

def calculate_sent_ndvi(image_file):
    image,_,profile,_ = openRaster(image_file)
    calc_img = eval(bp.sent_NDVI, bp.sent_bands_dict)
    profile.update({'count':1}) 
    with rio.open(calc_img, 'w', **profile) as dst:
        dst.write(calc_img, 1)

