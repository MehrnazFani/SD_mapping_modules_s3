import rasterio as rio
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from rasterio import features
from collections import OrderedDict
from datetime import datetime
import sys
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from PIL import Image
from io import BytesIO
import geopandas as gpd
import pickle


s3 = boto3.client('s3')

class UseBucket:
    
    def __init__(self, bucket_name="geomate-data-repo-dev"):
        self.bucket_name = bucket_name
    
    def s3_listdir( self, directory, suffix=""):
        "mimic os.listdir for s3 bucket, and even go beyond"
        # List objects within the directory specified in prefix, having the type specified in suffix.
        # if no suffix is provided all the files in the prefix directory will be listed
        # for example for listing this file 'bucket_name/path/to/file/image1.jp2   
        # prefix= directory = 'path/to/files/'    
        # suffix = '.jp2'

        # Initialize the paginator
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=directory)

        file_keys = []
        
        # Iterate through each page of results and get all the file_keys for files ending with suffix
        for page in page_iterator:
            file_keys_page = []
            if 'Contents' in page:
                file_keys_page = [obj['Key'] for obj in page['Contents'] if obj['Key'].lower().endswith((suffix))]
                file_keys.extend(file_keys_page)
        file_names = [fkey.split('/')[-1] for fkey in file_keys]
        if  not file_names:
                print(f"No file of type {suffix} is found in this directory!") 
        return file_names

    
    def s3_glob(self, directory, suffix=""):
        "mimic glob.glob() for s3 bucket, and even go beyond"
        # List full dir of objects within the directory specified in prefix, having the type specified in suffix.
        # if no suffix is provided all the files in the prefix directory will be listed
        # for example for listing this file 'bucket_name/path/to/file/image1.jp2   
        # prefix='path/to/file/'    
        # suffix = '.jp2'
        
        # Initialize the paginator
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=directory)

        file_keys = []
        
        # Iterate through each page of results and get all the file_keys for files ending with suffix
        for page in page_iterator:
            file_keys_page = []
            if 'Contents' in page:
                file_keys_page = [obj['Key'] for obj in page['Contents'] if obj['Key'].lower().endswith((suffix))]
                file_keys.extend(file_keys_page)
        if  not file_keys:
            print(f"No file of type {suffix} is found in this directory!") 
        return file_keys
    
   
    def s3_walk(self, prefix=''):
        """Mimics os.walk for an S3 bucket, yielding (directory, subdirs, files)."""
        paginator = s3.get_paginator('list_objects_v2')
        prefix = prefix.rstrip('/') + '/' if prefix else ''
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix, Delimiter='/'):
            # Current "directory"
            current_dir = page.get('Prefix', '')
            
            # Subdirectories in this "directory"
            subdirs = [prefix['Prefix'] for prefix in page.get('CommonPrefixes', [])]
            
            # Files in this "directory"
            files = [content['Key'].split('/')[-1] for content in page.get('Contents', []) if content['Key'] != current_dir]
            
            yield current_dir, subdirs, files

    def s3_dir_exists(self, prefix):
        "mimic os.path.exist('dir') for s3 bucket and a dir = prefix"    
        # Ensure the prefix ends with '/' to treat it like a directory
        prefix = prefix.rstrip('/') + '/'
        response = s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix, MaxKeys=1)
        return 'Contents' in response  # True if the prefix exists, False otherwise
    
    def s3_file_exists(self, key):
        """
        mimics os.path.exist('file_path') for a file on s3 bucket key = file_path
        Check if a file exists in an S3 bucket.
        param key: Path of the file (object key) in the bucket
        return: True if the file exists, False otherwise
        """
        try:
            s3.head_object(Bucket=self.bucket_name, Key=key)
            return True  # If head_object succeeds, the file exists
        except ClientError as e:
            # Check for "Not Found" error (404)
            if e.response['Error']['Code'] == "404":
                return False
            else:
                # For other errors, re-raise the exception
                raise
    
    def s3_makedirs(self, folder_path):
        if not folder_path.endswith('/'):# Ensure folder path ends with a slash
            folder_path += '/'
        # Create a zero-byte object with the folder path
        s3.put_object(Bucket=self.bucket_name, Key=folder_path) 

    def s3_gpd_gs_read_file(self, s3_key):  
        "Mimics the following steps, while s3_key = file_adrs"    
        # file_adrs is path/to/file.geojson       
        # gs = gpd.GeoSeries.from_file(file_adrs)        
        response = s3.get_object(Bucket=self.bucket_name, Key=s3_key)  # Fetch the file from S3 into memory
        file_data = response['Body'].read()  # Read the content of the file into memory
        file_like_object = BytesIO(file_data) # Use io.BytesIO to treat the in-memory content as a file-like object
        gs = gpd.read_file(file_like_object).geometry # Read the GeoJSON (or other vector format) directly into a GeoSeries
        return gs
    
    def s3_rasterio_open(self, s3_key):
        "Mimics src = rasterio.open(img_adrs), while s3_key = image_adrs"
        s3_url = f"s3://{self.bucket_name}/{s3_key}"

        from rasterio.errors import RasterioIOError
        attempts = 3  # Maximum number of attempts to open the image
        for attempt in range(1, attempts + 1):
            try:
                # Try opening the JP2 image using rasterio
                src = rio.open(s3_url)
                return src
                break
            except RasterioIOError as e:
                print(f"RasterioIOError: {e}")
                if attempt == max_attempts:
                    print("All attempts failed. Re-raising the exception.")
                    raise e  # Re-raise the exception if all attempts fail
        
    def s3_df_to_csv(self, data_frame, output_path):
        "Mimics df.to_csv('output_path')"
        # Save DataFrame to an in-memory binary buffer
        csv_buffer = BytesIO()
        data_frame.to_csv(csv_buffer)  # Write CSV data to the buffer
        csv_buffer.seek(0)  # Reset the buffer's cursor to the beginning
        # output_path = s3_key  within the S3 bucket
        try:
            s3.put_object( Bucket=self.bucket_name, Key=output_path, Body=csv_buffer)
            print(f"File uploaded to s3://{self.bucket_name}/{output_path}")
        except Exception as e:

            print(f"Error uploading file: {e}")
 

    def s3_write_txt(self, output, output_path): 
        """
        Mimics The Following:
            with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)
        """  
        s3_key = output_path # The path and filename in the S3 bucket
        # output Content to upload
        # Write to S3
        s3.put_object(Body=output, Bucket=self.bucket_name, Key=s3_key)
        print(f"File successfully written to s3://{self.bucket_name}/{s3_key}")

    
    def s3_pickle_dump(self, path_to_save_rasters, temp_raster):
        """ Mimic 
            with open(path_to_save_rasters, 'wb') as f:
                pickle.dump(temp_raster, f)
        """
        # Serialize the data to a binary format
        serialized_data = pickle.dumps(temp_raster)
        
        # Write the binary data directly to S3
        object_key = path_to_save_rasters
        s3.put_object(Bucket=self.bucket_name, Key=object_key, Body=serialized_data)
    
    def s3_torch_load(self, device, path_to_model):

        model_key = path_to_model  # S3 path to the checkpoint exlude s3://bucket_name/

        # Stream the file from S3
        response = s3.get_object(Bucket=self.bucket_name, Key=model_key)
        checkpoint_data = response['Body'].read()

        # Load the checkpoint directly from memory
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(BytesIO(checkpoint_data), map_location=device)

        # Use the checkpoint
        print("Model Checkpoint Loaded Successfully!")
        return checkpoint
    

    # def s3_torch_load(self,device, path_to_model, temp_local_model_path):
    #     # Define the S3 bucket and file details
    #     model_key = path_to_model # S3 path to the checkpoint
    #     local_model_path = temp_local_model_path  # Temporary local path to save the checkpoint

    #     # Download the file from S3
    #     s3.download_file(self.bucket_name, model_key, local_model_path)

    #     # Load the checkpoint
    #     checkpoint = torch.load(local_model_path, map_location=device)

    #     # Optional: Clean up the temporary file after loading
    #     #os.remove(local_model_path)

    #     # Use the checkpoint
    #     print("Checkpoint loaded successfully!")
    #     return checkpoint
    