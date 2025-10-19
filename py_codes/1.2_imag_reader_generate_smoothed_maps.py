# -*- coding: utf-8 -*-

"""

Time interpolation of each pixel of the TIFF images (no spatial smoothig in this step)

"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import sys
import os
import glob

from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

from scipy.sparse import diags, csc_matrix
from scipy.linalg import cho_factor, cho_solve

### Configuration area

lambda_whittaker = 100

### end of Configuration


whittaker_ = True


path = r"../data/all_imags_raw" + os.sep
output_path_bsi  = r"../data/interpolated_images/bsi" + os.sep
output_path_ndwi = r"../data/interpolated_images/ndwi" + os.sep



remove_sce_labels = [1, 2, 3, 8, 9, 10]



"""
Band and idexes

Band 01: B1         ind 0
Band 02: B2         ind 1
Band 03: B3         ind 2
Band 04: B4         ind 3
Band 05: B5         ind 4
Band 06: B6         ind 5
Band 07: B7         ind 6
Band 08: B8         ind 7
Band 09: B8A        ind 8
Band 10: B9         ind 9
Band 11: B11        ind 10
Band 12: B12        ind 11
Band 13: ndvi       ind 12
Band 14: ratio83    ind 13
Band 15: ratio211   ind 14
"""



NODATA_VALUE = -9999.0



# ---- Read Image and Processing Functions ----
def read_pic(tif_file):
    """ Reads a TIFF file using rasterio """
    with rasterio.open(tif_file) as f:
        bands = f.read()
        meta = f.meta  # Save metadata for output
    return bands, meta

def remove_bad_pixels(bands_imags, filter_imag, class_vector):
    """ Filters out unwanted pixels using classification image """
    imags = bands_imags
    output_imags = np.zeros(imags.shape)

    for i in range(len(imags)):
        mask = np.isin(filter_imag, list(class_vector))
        output_imags[i] = np.where(mask, np.nan, imags[i])
    return output_imags

def whittaker_smooth(x, lmbd):
    m = len(x)
    E = np.eye(m)
    D = diags([1, -2, 1], [0, 1, 2], shape=(m-2, m)).toarray()
    A = E + lmbd * (D.T @ D)
    c, low = cho_factor(A)
    return cho_solve((c, low), x)



def calculate_bsi(B11, B4, B8, B2, epsilon=1e-10):
    """
    Bare Soil Index (BSI)
    """
    return ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2) + epsilon)

def calculate_ndwi(b8, b11, epsilon=1e-10):
    """
    Calculate the ndwi index
    NDWI = (NIR - SWIR1)/(NIR + SWIR1)
    """
    return (b8 - b11)/(b8 + b11 + epsilon)


# ---- Path and File Management ----

os.makedirs(output_path_bsi,  exist_ok=True)
os.makedirs(output_path_ndwi,  exist_ok=True)


# ---- Read Images and Extract NDVI Time Series ----
f_names = sorted(glob.glob(f"{path}/*.tif"))
pairs = [f_names[i:i + 2] for i in range(0, len(f_names), 2)]

valid_dates = []
filtered_data_output = []
bsi_values  = []
ndwi_values = []


for pair in pairs:
    data_imag, meta = read_pic(  pair[0]  )
    filter_imag, _ = read_pic(   pair[1]  )
    
    filtered_imag = remove_bad_pixels(data_imag, filter_imag[0], remove_sce_labels)
    date_imag = datetime.strptime(pair[0][-14:-4], "%Y-%m-%d").date()   #  IMPORTANT; change index if the name chnages
    
    valid_dates.append(date_imag)
    filtered_data_output.append([date_imag, filtered_imag])
    
    
    ## Calc indexes
    filtered_bsi  = calculate_bsi(   filtered_imag[10], filtered_imag[3], filtered_imag[7], filtered_imag[1])
    bsi_values.append([  date_imag, filtered_bsi ])
    
    filtered_ndwi     = calculate_ndwi(       filtered_imag[7],filtered_imag[10]  )
    ndwi_values.append([     date_imag,filtered_ndwi     ])


# ---- Convert Dates to Ordinal Numbers ----
dates_numeric = np.array([d.toordinal() for d in valid_dates])

# ---- Get Image Shape for Reconstruction ----
image_shape = filtered_data_output[0][1].shape  # (bands, rows, cols)
_, rows, cols = image_shape

# ---- Interpolate NDVI for Each Pixel ----
full_dates_numeric = np.arange(dates_numeric[0], dates_numeric[-1] + 1)
full_dates = [datetime.fromordinal(int(d)) for d in full_dates_numeric]

# Create an array to store interpolated images
interpolated_bsi_images  = np.zeros((len(full_dates), rows, cols), dtype=np.float32)
interpolated_ndwi_images    = np.zeros((len(full_dates), rows, cols), dtype=np.float32)


## speed up Whittaker
m = len(full_dates_numeric)
E = np.eye(m)
D = diags([1, -2, 1], [0, 1, 2], shape=(m-2, m))
DTD = D.T @ D
A = E + lambda_whittaker * DTD
c_factor = cho_factor(A)


for i in range(rows):
    #if (i % 10) == 0: print(f"run: {i+1}")
    print(f"run: {i+1}")
    for j in range(cols):
        # Extract NDVI time series for pixel (i, j)
        bsi_values_px  = np.array([ bsi_values[time][1][i][j] for time in range(len(bsi_values))])
        ndwi_values_px    = np.array([ ndwi_values[time][1][i][j] for time in range(len(ndwi_values))])
        
        # Remove NaN values for interpolation
        valid_mask = ~np.isnan(bsi_values_px)
        valid_dates_numeric = dates_numeric[  valid_mask]
        valid_bsi_values  = bsi_values_px[    valid_mask]
        valid_ndwi_values    = ndwi_values_px[    valid_mask]
        
        
        if whittaker_:
            if len(valid_dates_numeric) > 5:
                # --- Whittaker Smoothing ---
                # Interpolate linearly to create daily values
                
                linear_interp = interp1d(valid_dates_numeric, valid_bsi_values, kind='linear', fill_value="extrapolate")
                linear_interpolated = linear_interp(full_dates_numeric)
                interpolated_bsi_values     = cho_solve(c_factor,  linear_interpolated)
                
                linear_interp = interp1d(valid_dates_numeric, valid_ndwi_values, kind='linear', fill_value="extrapolate")
                linear_interpolated = linear_interp(full_dates_numeric)
                interpolated_ndwi_values    = cho_solve(c_factor,  linear_interpolated)
                
            else:
                # Fallback: all NaNs if not enough data
                interpolated_bsi_values  = np.full_like(full_dates_numeric, np.nan, dtype=np.float32)
                interpolated_ndwi_values    = np.full_like(full_dates_numeric, np.nan, dtype=np.float32)
        
        # Store interpolated values
        interpolated_bsi_images[:, i, j]  = interpolated_bsi_values
        interpolated_ndwi_images[:, i, j]     = interpolated_ndwi_values



# ---- Save Interpolated data Images as GeoTIFF ----
for idx, date in enumerate(full_dates):
    output_bsi_filename  = os.path.join(output_path_bsi,  f"bsi_{date.strftime('%Y-%m-%d')}.tif")
    output_ndwi_filename     = os.path.join(output_path_ndwi,     f"ndwi_{date.strftime('%Y-%m-%d')}.tif")
    
    
    # Copy metadata and update necessary fields
    new_meta = meta.copy()
    new_meta.update({
        "count": 1,         # Only saving one band (NDVI)
        "dtype": "float32",  # Ensure float32 format
        "nodata": NODATA_VALUE  # Explicitly mark NoData value
    })
    
    # Create a copy of the image data
    image_bsi_data  = interpolated_bsi_images[idx, :, :].copy()
    image_ndwi_data     = interpolated_ndwi_images[idx, :, :].copy()
    
    # Ensure NoData is set correctly
    nan_mask = np.isnan(image_bsi_data)
    image_bsi_data[nan_mask] = NODATA_VALUE
    
    nan_mask = np.isnan(image_ndwi_data)
    image_ndwi_data[nan_mask] = NODATA_VALUE
    
    if False:
        # filtering values, max and min
        image_bsi_data[image_bsi_data < -1]   = -1
        image_bsi_data[image_bsi_data > +1]   = +1
        
        image_ndwi_data[image_ndwi_data < -1]   = -1
        image_ndwi_data[image_ndwi_data > +1]   = +1
    
    
    # Save the raster
    with rasterio.open(output_bsi_filename, "w", **new_meta) as dst:
        dst.write(image_bsi_data, 1)  # Write single band
    
    
    with rasterio.open(output_ndwi_filename, "w", **new_meta) as dst:
        dst.write(image_ndwi_data, 1)  # Write single band
    
    if (idx % 10) == 0: print(f"index {idx} of {len(full_dates)}")


print("All interpolated images saved.")






sys.exit()





