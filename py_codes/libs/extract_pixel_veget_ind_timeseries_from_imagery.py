# -*- coding: utf-8 -*-

"""

Extract the timeseries, for a pixel, from imagery


"""



import os
import glob
import sys

import rasterio

import numpy as np
from datetime import datetime, timedelta
import pandas as pd




# first and last days included
initial_date = "2020-09-16"
final_date   = "2021-08-24"


indxs_image = [[52,31],
               [52,32],
               [52,34]]


path = r"D:\_Trabalho\_Publicacoes_nossas\Learning_Material_Workshop\material_Boo\training_12_pts\sat_imag\interpolated_images\_gauss_filt\ndvi" + os.sep

output_path = r"D:\_Trabalho\_Publicacoes_nossas\Learning_Material_Workshop\material_Boo\training_12_pts\sat_imag\interpolated_images\_gauss_filt\time_series" + os.sep
output_file_name = output_path + "output.csv"

# save output to file?
save = False

# interval to find the date in the file name (Dont change!!!)
interval = [-14,-4]


def pix_timeseries(indxs_image, initial_date, final_date, path, interval=interval, save = False, output_file_name=output_file_name):
    
    """
    indxs_image : indexes of the desired pixels
    initial_date : initial date
    final_date : end date
    path : path to the folder containing the tiff files
    interval : interval in the file name containig the date
    save : Bool, save the file?
    output_file_name : path + name of the file, if save == True.
    
    
    it returns (dates , pixel_values)
    
    Structures
    dates = string, YYYY-MM-DD
    pixel_values = numpy matrix[pixel][values through time] = pixel_value and
    
    """
    
    ini_date = datetime.strptime(initial_date,  "%Y-%m-%d")
    end_date = datetime.strptime(final_date, "%Y-%m-%d")
    
    # Get sorted list of all TIFF images
    f_names = sorted(glob.glob(f"{path}/*.tif"))
    
    # image stack with zeros
    num_images = abs((end_date-ini_date).days) + int(1)
    num_pixels = len(indxs_image)
    
    stack_data = np.zeros((num_pixels , num_images))
    stack_dates = [[] for _ in range(num_images)]
    
    idx = int(0)
    for file in f_names:
        with rasterio.open(file) as f:
            actual_date = datetime.strptime( file[ interval[0] : interval[1] ],  "%Y-%m-%d")
            
            if (actual_date >= ini_date) and (actual_date <= end_date):
                bands = f.read(1)
                stack_dates[idx] = actual_date.strftime('%Y-%m-%d')
                
                for ind, value in enumerate(indxs_image):
                    imag_temp = bands[value[0] , value[1]]
                    # numpy timeseries
                    stack_data[ind][idx] = imag_temp
                idx += int(1)
    
    
    return (stack_dates, stack_data)


## TESTS!
if __name__ == "__main__":
    aa1, aa2 = pix_timeseries(indxs_image, initial_date, final_date, path)
    
    dates = aa1
    vals  = aa2
    for i in range(len(aa1)):
        print(dates[i], vals[0][i], vals[1][i])
    
    print(vals.shape)








