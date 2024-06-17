from ImageOperations import RasterOperations as ro 
from DataPreparation import DataFromVectors as dv
from pathlib import Path
image_var_name = 'raster'

image_batch_dict = dict()
# dictionary type {'raster_1': {'inp_files':[], 'out_file': ''}
class ImageBatchKeys:
    def __init__(self,):
        self.input_data = 'input_data'
        self.parent_folder = 'parent_folder'
        self.stacked_img = 'stacked_image'
        self.files_list = 'files_list'
        self.date = 'date'
        self.clipped_files = 'clipped_files'
        self.calculated_files = 'calculated_files'
        self.mosaicked_file = 'mosaicked_file'
        self.stacked_img = 'stacked_image'
def setup_image_dict():
    image_super_dict = {'input_data': dict(),
                        'parent_folder': Path('.'),
                        'stacked_image': Path('stacked.tif'),
                        }
    return image_super_dict
def get_image_sub_dict(files_list, out_date):
    '''
    Get image sub dictionary
    '''
    image_dict = dict()
    image_dict['files_list'] = files_list
    image_dict['date'] = out_date
    image_dict['clipped_files'] = []
    image_dict['mosaicked_file'] = ''
    parent_folder = Path(files_list[0]).parent
    return image_dict, parent_folder

def getoradd_image_dict(image_super_dict, files_list, out_date):
    if len(image_super_dict['input_data'].keys()) == 0:
        key_name = f'{image_var_name}_1'
    else:
        success, last_idx, reason = check_nomenclature_consistency(image_super_dict=image_super_dict)
        if success == True:
            key_name = f'{image_var_name}_{last_idx}'
        else:
            print(reason)
    image_super_dict['input_data'][key_name], \
    image_super_dict['parent_folder']= \
        get_image_sub_dict(files_list=files_list, out_date=out_date)

def check_nomenclature_consistency(image_super_dict):
    '''
    This framework expects the keys of the super dictionary to be image_var_name
    followed by '_' then number starting with 1
    If the keys do not follow the nomenclature, further issues will be created
    '''
    keys = image_super_dict['input_data'].keys()
    success = True
    last_idx = 1
    for i in keys:
        test = i.split('_')
        if len(test) == 2 and test[0]==image_var_name:
            continue
        else:
            reason = f'{i} in the keys is odd'
            success = True
            return (success, None, reason)
        try: 
            int(test[1])
            last_idx = max(last_idx, int(test[1]))
        except:
            reason = f'{i} in the keys has non int index'
            success = True
            return (success, reason)
    if success == True: 
        return (success, last_idx, 'Dictionary compliant with nomenclature')
    else:
        return (success, None, 'Unidentified error')

