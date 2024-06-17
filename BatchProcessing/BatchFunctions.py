from DataPreparation import DataFromVectors as dv
from ImageOperations import RasterOperations as ro
from ImageOperations import BandProfile as bp
from BatchProcessing import ImageBatchSetup as ib
import rasterio as rio
from pathlib import Path
# Setup image list
# Clip
# Mosaic
# Stack

class StandardSuffixes:
    def __init__(self,):
        self.clipped = 'clipped'
        self.calculated = 'calc'
        self.mosaicked = 'mosaic'
        self.stack_name = 'stacked'

#Setup Variable Names

suffix = StandardSuffixes()
ib_keys = ib.ImageBatchKeys()

def get_file_name(folder_path, name_list, ext):
    name = ''
    for char in name_list:
        if len(name) > 0:
            name += '_'
        name += char
        file_name = folder_path / (name+ext)
    return file_name

def batch_clip(files_list, shape_file):
    clipped_image_list = list()
    for file in files_list:
        file_name = Path(file).stem
        ext = Path(file).suffix
        name_list = [file_name, 'clipped']
        temp_out_file = get_file_name(root_folder,
                                      name_list=name_list,
                                      ext=ext)
        
        clipped = ro.clipFromVector(file,
                                      shape_file,
                                      temp_out_file)
        if len(clipped) > 1:
            print ('Error: More that one shape in shape file')
            break
        clipped_image_list.append(clipped[0])
    return clipped_image_list

def batch_mosaic(files_list, suffix_name, root_folder, **kwargs):
    ext = Path(files_list[0]).suffix
    temp_out_file = get_file_name(root_folder,
                                  name_list=[suffix_name],
                                  ext = ext)
    return ro.mosaicRaster(files_list, temp_out_file)

def batch_band_calc(file_list, raster_func_list,
                    root_folder,
                    add_new_layer=True,
                    band_dict=bp.sent_bands_dict,
                    suffix_name=suffix.calculated):
    out_list = []
    for image_file in file_list:
        file_name = Path(image_file).stem
        ext = Path(image_file).suffix
        name_list = [file_name,suffix_name]
        out_file_name = get_file_name(root_folder,
                                      name_list=name_list,
                                      ext=ext)
        
        ro.calculate_func(image_file=image_file,
                                    out_file=out_file_name,
                                    raster_func=raster_func_list,
                                    add_new_layer=add_new_layer,
                                    band_dict=band_dict)
        out_list.append(out_file_name)
    return out_list

def batch_stack(image_batch_dict, key_to_stack,
                root_folder, stack_name):
    #Check the image_batch_dict nomenclature
    stack_list = []
    for key in image_batch_dict.keys():
        stack_list.append(image_batch_dict[key][key_to_stack])
    name_list = [stack_name]
    out_file = get_file_name(folder_path=root_folder,
                             name_list=name_list) 
    ro.stackImages(files_list=stack_list,
                   out_file=out_file,)
    image_batch_dict.update({ib_keys.stacked_img: out_file})
    return image_batch_dict


class BatchImageProcess:
    def __init__(self, image_batch_file, shape_file_path, band_dict, raster_func_list,
                 add_new_layer=False):
        self.image_batch = image_batch_file
        self.parent_folder = image_batch_file[ib_keys.parent_folder]
        self.input_dict = image_batch_file[ib_keys.input_data]
        self.shape_file_path = shape_file_path
        self.raster_func_list = raster_func_list
        self.band_dict = band_dict
        self.add_new_layer = add_new_layer
    def batch_clip(self,raster_key):
        files_list = self.input_dict[raster_key][ib_keys.files_list]
        return batch_clip(files_list, shape_file=self.shape_file_path)
    def batch_band_calc(self,
                        raster_key,
                        suffix_name = suffix.calculated):
        file_list = self.input_dict[raster_key][ib_keys.clipped_files]
        return batch_band_calc(
                    file_list=file_list,
                    raster_func_list=self.raster_func_list,
                    add_new_layer=self.add_new_layer,
                    band_dict= self.band_dict,
                    suffix_name = suffix_name)

    def batch_mosaic(self,files_list, suffix_name=suffix.mosaicked):
         return batch_mosaic(files_list, suffix_name)
    def batch_stack(self,
            key_to_stack= ib_keys.mosaicked_file,
            stack_name=suffix.stack_name):
         return batch_stack(image_batch_dict=self.input_dict,
                               key_to_stack=key_to_stack,
                               stack_name=stack_name)

    def Sentinel_Stack01(self,):
        for key in self.input_dict.keys():
            # Raster_1 type keys
            files_list = self.input_dict[key][ib_keys.files_list]
            self.input_dict[key][ib_keys.clipped_files] = self.batch_clip(raster_key = key)
            self.input_dict[key][ib_keys.calculated_files] = self.batch_band_calc()
            self.input_dict[key][ib_keys.mosaicked_file] = self.batch_mosaic(
                files_list=self.input_dict[key][ib_keys.calculated_files],
                add_name=self.input_dict[key][ib_keys.date])
        # Stack all Images
        self.image_batch[ib_keys.stacked_img] = self.batch_stack()
