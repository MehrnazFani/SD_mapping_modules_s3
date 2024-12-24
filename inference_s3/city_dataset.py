import collections
import math
import os
import random
#import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.windows import Window
import csv
import PIL

from data_utils import affinity_utils

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms

import pandas as pd
import glob
import copy
from s3_utils import UseBucket
bucket_name = "geomate-data-repo-dev"
my_bucket = UseBucket(bucket_name)

class testCityDataset(Dataset):
    def __init__(self, path_to_tiles):
        self.path_tiles = path_to_tiles
        #self.tile_names = glob.glob(os.path.join(path_to_tiles, '*.jp2'))
        self.tile_names = my_bucket.s3_glob(path_to_tiles, suffix = ".jp2")
        # print (self.tile_names)

    def __len__(self):
        return len(self.tile_names)

    def transforms(self, image):
        _image = TF.to_pil_image(torch.tensor(image))
        _image = TF.resize(_image, size=768, interpolation=PIL.Image.NEAREST)
        _image = TF.to_tensor(_image)

        return _image

    def morph_meta(self, meta):
        _meta = meta

        affine = _meta['transform']
        affine_trans = [affine.a, affine.b, affine.c, affine.d, affine.e, affine.f, ]

        _meta.pop('transform')  # removed as it might crash the program, might be fixed using a custom collate.
        _meta.pop('nodata')  # removed as it might crash the program, might be fixed using a custom collate.
        # print (_meta)
        _meta.pop('crs')

        meta_dict = {'_meta': _meta,
                     'transform': affine_trans}

        return meta_dict

    def __getitem__(self, index):
        # print (self.tile_names[index])
        #src = rio.open(self.tile_names[index])
        src = my_bucket.s3_rasterio_open(self.tile_names[index])
        name = self.tile_names[index].split('/')[-1]
        # print (name)
        try:
            image = src.read(indexes=(1, 2, 3))

            image = self.transforms(image)
            # print(src.meta)
            meta_dict = self.morph_meta(src.meta)
            output = [image, meta_dict, name]

        except:
            output = [[], [], name]
        # print (meta_dict)
        # print (f'{image}, {type(image)}')
        # print (f'{meta_dict}, {type(meta_dict)}')
        # print (f'{name}, {type(name)}')
        return output
