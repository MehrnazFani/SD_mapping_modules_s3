import rasterio as rio
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from tqdm import tqdm
from rasterio import features
import pickle
import multiprocessing
from model.models import MODELS
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from city_dataset import testCityDataset
from collections import OrderedDict
import glob
import argparse
import json
from datetime import datetime
import sys
import boto3
from s3_utils import UseBucket

bucket_name = "geomate-data-repo-dev"
my_bucket = UseBucket(bucket_name)

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="config file path", default="config.json")


def extract_time ():
    now = str(datetime.now()).replace("-", "_").replace(" ", os.path.normpath("/"))
    return now[:str(now).rfind(":")].replace(":", "_")

    
class Inference:
    def __init__(self, 
                 path_to_tiles, 
                 path_to_model, 
                 path_to_save_rasters, 
                 batch_size, 
                 num_workers,
                 model_trained_on_multiple_gpu):

        self.path_to_tiles = path_to_tiles
        self.path_to_model = path_to_model
        self.path_to_save_rasters = path_to_save_rasters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_trained_on_multiple_gpu = model_trained_on_multiple_gpu
        self.new_state_dict = OrderedDict()
        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        if path_to_model[0] !='/': # s3_key for weights does not strart with /
            # Loading the mode from s3 if path_to model is an s3_key
            self.checkpoint = my_bucket.s3_torch_load(self.device, path_to_model)
        else:
            # Loading the mode from local if path_to model is local
            self.checkpoint = torch.load(path_to_model, map_location=self.device)
        
        self.model = MODELS["StackHourglassNetMTL"]()

    def load_state_dict(self):
        """
        original saved file with DataParallel
        create new OrderedDict that does not contain `module.`
        :param self:
        :return: None (modify self.new_state_dict inplace)
        """
        # original saved file with DataParallel
        # create new OrderedDict that does not contain `module.`

        for k, v in self.checkpoint['state_dict'].items():
            name = k.split("module.")[-1]
            self.new_state_dict[name] = v

    def load_params(self):
        """
        load params and switch to the model to eval mode
        :return: None (modify self.model inplace)
        """
        # load params
        if self.model_trained_on_multiple_gpu:
            self.model.load_state_dict(self.new_state_dict)
        else:
            self.model.load_state_dict(self.checkpoint['state_dict'])
    
        self.model.eval()
        self.model.to(self.device)
    
    def build_test_loader(self):
        """
        Building test loader
        :return: test_loader
        """
        test_ds = testCityDataset(self.path_to_tiles)

        test_loader = DataLoader(test_ds, 
                                batch_size=self.batch_size,     
                                shuffle=False, 
                                num_workers=self.num_workers,
                                pin_memory=True)
        return test_loader

def inference_driver(datasets_name,
                     cities, 
                     tile_size,
                     overlap_size,
                     project_name,
                     path_to_model, 
                     batch_size,
                     num_workers,
                     model_trained_on_multiple_gpu,
                     user_name):

    print("cities:", cities)
    cities = [ x for x in str.split(cities, "_") if len(x)>0]
    inputset = {}
    now = extract_time ()

    list_of_tiles_with_error = {}

    for city_info in cities:
        info = [str(i).replace("[", "").replace("]", "") for i in str(city_info).split(",")]
        print("info",info)

        month = ""
        if len(info) == 2:
            [city,  percentage] = info

        elif len(info) == 3:
            [city,  percentage, month] = info

        dir_files = f"{datasets_name}"

        if len(month) > 0:
            path_to_tiles = os.path.join(dir_files, city, 'city-images', month, 'tiles', percentage + '_percent', str(tile_size) + "_" + str(overlap_size))
        else:
            path_to_tiles = os.path.join(dir_files, city, 'city-images', 'tiles', percentage + '_percent', str(tile_size) + "_" + str(overlap_size))


        if not my_bucket.s3_dir_exists(path_to_tiles):
            print(f"{path_to_tiles} does not exist")
            sys.exit(1) # exit code because of this error

        if len(month) > 0:
            root_address = os.path.join(dir_files, city, 'inference', month, project_name, percentage + '_percent', str(tile_size) + "_" + str(overlap_size),  user_name, now).replace("ُ", "")
        else:
            root_address = os.path.join(dir_files, city, 'inference', project_name, percentage + '_percent', str(tile_size) + "_" + str(overlap_size),  user_name, now).replace("ُ", "")

        path_to_save_rasters = os.path.join(root_address, "outputs")
        list_of_tiles_with_error_for_each_city = \
        cover_single_city(path_to_tiles, 
                          path_to_model,
                          path_to_save_rasters, 
                          project_name,
                          batch_size,
                          num_workers,
                          model_trained_on_multiple_gpu,
                          **inputset)

        list_of_tiles_with_error[city] = list_of_tiles_with_error_for_each_city

    print("corrupted Tiles List:")
    print(json.dumps(list_of_tiles_with_error, indent=4))

def cover_single_city(path_to_tiles, 
                      path_to_model,
                      path_to_save_rasters, 
                      project_name,
                      batch_size,
                      num_workers,
                      model_trained_on_multiple_gpu,
                      **inputset):

    inferencer = Inference(path_to_tiles, 
                           path_to_model,
                           path_to_save_rasters, 
                           batch_size,
                           num_workers,
                           model_trained_on_multiple_gpu)
    
    tiles = my_bucket.s3_glob(inferencer.path_to_tiles, suffix=".jp2")
    tile_keys = tiles
    src_vrt = my_bucket.s3_rasterio_open(tile_keys[0])

    inferencer.load_state_dict()
    inferencer.load_params()
    inference_loader = inferencer.build_test_loader() 
    rasters = {}
    list_of_tiles_with_error_for_each_city = []

    for j, sample in tqdm(enumerate(inference_loader)):
        
        # Detecting the Corrupted Tiles
        if len(sample[0]) == 0:
            list_of_tiles_with_error_for_each_city.append(sample[2][0])
            continue

        else:
            names = sample[2]

        sys.stdout.write("\033[F")
        print (f"\nread sample {j}(/{len(inference_loader)-1}) - ({names[0]}) ")
        with torch.no_grad():
            inputRGB = sample[0].to(inferencer.device)
            output_set = inferencer.model(inputRGB)
        
            original_outputs = []
            original_pred_vecmaps = []
            for temp_index in range(0, len(output_set), 2):
                original_outputs.append(output_set[temp_index])
            for temp_index in range(1, len(output_set), 2):
                original_pred_vecmaps.append(output_set[temp_index])
            

            for index in range(len(original_outputs)):
                outputs = original_outputs[index]
                pred_vecmaps = original_pred_vecmaps[index]

                if len(original_outputs) > 1:
                    folder_name = f"class_{index+1}"
                
                else:
                    folder_name = ""

                for i in range(len(inputRGB)):
                    pred = (F.softmax(outputs[3].detach().cpu(), dim=1))[i,1,:,:]

                    _pred = np.stack([pred, pred, pred], axis = 0)
                    # _pred[_pred < (0.9 * np.max(_pred))] = 0
                    _pred = _pred * (1/np.max(_pred)) * 255
                    _pred = _pred.astype(np.uint8)
            
                    meta = dict()
                    meta['driver'] = 'GTiff'    # rasterio can write only with GTiff driver
                    meta['dtype'] = src_vrt.meta['dtype']
                    meta['nodata'] = src_vrt.meta['nodata']
                    meta['width'] = int(sample[1]['_meta']['width'][i])
                    meta['height'] = int(sample[1]['_meta']['height'][i])
                    meta['count'] = 3
                    meta['crs'] = src_vrt.meta['crs']

                    a = float(sample[1]['transform'][0][i])
                    b = float(sample[1]['transform'][1][i])
                    c = float(sample[1]['transform'][2][i])
                    d = float(sample[1]['transform'][3][i])
                    e = float(sample[1]['transform'][4][i])
                    f = float(sample[1]['transform'][5][i])
                
                    meta['transform'] = rio.Affine(a, b, c, d, e, f)
                
                    reference_size = ((abs(a)+abs(e))/2)*meta["width"]

                    _pred = resize_rasterio (_pred, meta)
 
                    vectorized_mask = vectorise(names[i], _pred, meta['transform'], meta["crs"], reference_size)

                    if folder_name in rasters.keys():
                        rasters[folder_name].append(vectorized_mask)

                    else:
                        rasters[folder_name] = [vectorized_mask]
                    

    manage_post_processing(rasters, inferencer.path_to_save_rasters)

    return list_of_tiles_with_error_for_each_city 

def resize_rasterio(_pred, meta):
    from rasterio.io import MemoryFile

    with MemoryFile() as memfile:
        with memfile.open(**meta) as dataset:
            dataset.write(_pred)
            del _pred
        with memfile.open() as dataset:  # Reopen as DatasetReader
            return dataset.read()[0]  

def vectorise(fname, red, transform, crs, width):
    red = np.intc(np.floor(np.divide(red,200)))
    mask = red == 1
    shapes = features.shapes(red, mask=mask,transform=transform)
    result = []
    for polygon in shapes:
        result.append({ "type": "Feature", "properties": {},"geometry": { "type": "Polygon", "coordinates":polygon[0]["coordinates"]}} )
    return [result,crs,round(width)]

def manage_post_processing(rasters, path_to_save_rasters):
    if not my_bucket.s3_dir_exists(path_to_save_rasters):
        my_bucket.s3_makedirs(path_to_save_rasters)
   


    for folder_name in rasters.keys():
        if len(folder_name) > 0: 
            path_to_save_rasters = os.path.join(path_to_save_rasters, folder_name, "rasters.pkl")

        else:
            path_to_save_rasters = os.path.join(path_to_save_rasters, "rasters.pkl")

        temp_raster = rasters[folder_name]
        print(f"len:{len(temp_raster)}")

        print(f"rasters.pkl is saved into: s3://{my_bucket}/{path_to_save_rasters}")
        my_bucket.s3_pickle_dump(path_to_save_rasters, temp_raster)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_ls', nargs="+", type=str, default= ['royal-oak-michigan-2021', 'birmingham-michigan-2021'], help='name of the city as mentioned in the mask full dir')
    parser.add_argument('--percent_ls', nargs="+", type=str, default=['100', '100'], help='name of the city as mentioned in the mask full dir')
    parser.add_argument('--month_ls', nargs="+", type=str, default=['mar', 'mar'], help='name of the city as mentioned in the mask full dir')
    return parser.parse_args()



if __name__ == '__main__':
    
    args = get_args()
    inference_cities = {}

    for city, percent, month in zip(args.city_ls, args.percent_ls, args.month_ls):
        inference_cities[city]=[percent, month]
    print(inference_cities)
    # input configuration
    config_dir = "./config.json"  
    #inference_cities = {'flint-michigan-2024':['intersection','und']}
    #{'troy-michigan-2024':['intersection','und'], 'flint-michigan-2024':['intersection','und'], 'auburnhills-michigan-sat-2024':['100','und'], 'southfield-michigan-maxar-2024':['100','und'] } 
    tile_size = "1250" 
    overlap_size = "150" 
    project_name = "RoadSurface_Detection"
    # path to model in s3 bucket, exclude s3://bucket_name/
    path_to_model = "weights/RoadSurface_NearMap_Maxar_general_model_20epochs.tar"
    #"/media/jerry/Data/weights/RoadSurface_TL_annarbor_8epochs.pth.tar"
    #"weights/RoadSurface_regina_TF_model_best_weights.pth.tar"
    user_name = "mehrnaz" 
    datasets_name = "datasets_vertical"
    batch_size = 1
    model_trained_on_multiple_gpu = True
    

    num_workers = multiprocessing.cpu_count()
  
    config = json.load(open(config_dir))
    cities = ""  
    for k,v in zip(inference_cities.keys(),inference_cities.values()):
        cities += f"_{k},{v[0]},{v[1]}"
    inference_driver(datasets_name,
                     cities, 
                     tile_size,
                     overlap_size,
                     project_name,
                     path_to_model, 
                     batch_size,
                     num_workers,
                     model_trained_on_multiple_gpu,
                     user_name)

# terminal run:
# python inference.py --city_ls 'royal-oak-michigan-2021' 'birmingham-michigan-2021' --percent_ls '100' '100' --month_ls 'mar' 'mar'