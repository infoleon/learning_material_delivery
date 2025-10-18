# -*- coding: utf-8 -*-
"""

Extract indexes (numpy images) from the selected coordinates

"""

import os
import glob
import sys
import rasterio
import numpy as np

import pandas as pd

from datetime import datetime, timedelta




def indexes_from_coordinates(f_name, coord_f_name, longit, latit, profile_id, print_ = False):
    
    """
    f_name : a image file containing the proper metadata 
    coord_f_name : file containig the selected coordinates
    longit : Longitude column name
    latit : Latitude column name
    profile_id: id of the profile (in the fild) column name
    """
    
    
    corrd_data = pd.read_csv(coord_f_name)
    ids_ = corrd_data[profile_id].unique().tolist()
    
    nested_corrd_data = corrd_data[[longit, latit]].values.tolist()
    
    # Which coordinates do we have in our TIFF image?
    with rasterio.open(f_name) as dataset:
        # Print the CRS of the TIFF
        if print_:
            print(f"TIFF CRS: {dataset.crs}")
        
        # Get the EPSG code
        epsg_code = dataset.crs.to_epsg()
        if print_:
            print(f"EPSG Code: {epsg_code}")
    
    
    # Open the TIFF file
    lon_t = -999
    lat_t = -999
    
    indexes = []
    with rasterio.open(f_name) as dataset:
        # Define your point of interest (Longitude, Latitude)
        
        for i in nested_corrd_data:
            lon_ = i[0]
            lat_ = i[1]
            
            #lon_ = 463375
            #lat_ = 5804905
            
            row, col = dataset.index(lon_, lat_)
            pixel_value = dataset.read(1)[row, col]
            
            # filtering repeated lat lon
            
            if print_:
                print(f"Point ({lon_}, {lat_}) corresponds to pixel indices: Row={row}, Col={col}")
            indexes.append([row, col])
            #print(f"Pixel value at this location: {pixel_value}")
            
            lon_t = lon_
            lat_t = lat_
    
    unique = []
    seen = set()
    for pair in indexes:
        tpl = tuple(pair)
        if tpl not in seen:
            seen.add(tpl)
            unique.append(pair)
    
    if print_:
        print()
    return unique
    #return indexes



if __name__ == "__main__":
    # Coordinate column names
    longit = "Easting"
    latit = "Northing"

    input_path_imag = r"D:\_Trabalho\_Publicacoes_nossas\Learning_Material_Workshop\material_Boo\training_12_pts\sat_imag\interpolated_images\ndvi" + os.sep
    f_name = input_path_imag + r"ndvi_2019-01-14.tif"

    coord_path = r"D:\_Trabalho\_Publicacoes_nossas\Learning_Material_Workshop\material_Boo\training_12_pts\monica_files\selection_GIS" + os.sep
    coord_f_name = coord_path + r"selection_GIS_32633_.csv"
    
    indexes_from_coordinates(f_name, coord_f_name, longit, latit, "Field_Profile_number")









