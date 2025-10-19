# -*- coding: utf-8 -*-


"""
Spatiotemporal smoothing of the indices time series (daily)
Images as TIFF

"""


import os
import glob
import sys
import rasterio
import numpy as np

from scipy.ndimage import gaussian_filter

### Configuration area

# Smoothing fators, temporal and spatial
sigma_t = 0.0
sigma_x = 1.0
sigma_y = sigma_x

### end of Configuration



# Which dataset?
bsi     = False
ndwi    = True





if bsi:
    input_path  = r"../data/interpolated_images/bsi" + os.sep
    ext = "bsi"
    out_path = r"../data/interpolated_images/_gauss_filt" + os.sep + ext + os.sep

if ndwi:
    input_path  = r"../data/interpolated_images/ndwi" + os.sep
    ext = "ndwi"
    out_path = r"../data/interpolated_images/_gauss_filt" + os.sep + ext + os.sep


os.makedirs(out_path, exist_ok=True)


# FUNCTIONS ####################################################################
def read_pic(tif_file):
    """ Reads a single-band TIFF file using rasterio """
    with rasterio.open(tif_file) as f:
        bands = f.read(1)
    return bands
def save_pic(out_file, stack):
    """ Save the pic """
    with rasterio.open(out_file, "w", **meta) as f:
        f.write(stack, 1)
def nan_gaussian_filter3d(data, sigma):
    """
    Apply Gaussian filter to 3D data with NaN support.
    data: 3D array (time, height, width)
    sigma: tuple (sigma_t, sigma_x, sigma_y)
    """
    nan_mask = np.isnan(data)
    data_filled = np.nan_to_num(data, nan=0.0)

    smoothed_data = gaussian_filter(data_filled, sigma=sigma)
    weights = gaussian_filter((~nan_mask).astype(float), sigma=sigma)

    with np.errstate(invalid='ignore', divide='ignore'):
        result = smoothed_data / weights
        result[weights == 0] = np.nan

    return result
# END OF FUNCTIONS #############################################################


# Get sorted list of all TIFF images
f_names = sorted(glob.glob(f"{input_path}/*.tif"))


# Taking height, width and number of images
with rasterio.open(f_names[0]) as src:
    meta = src.meta.copy()
    meta.update({"dtype": "float32"})  # Ensure float output
    height, width = src.shape
num_images = len(f_names)

print("... Reading and stacking images")

# creating zero matrice for the image stack (it is fater this way)
# then open files and stack data as Numpy matrices and dates

image_stack = np.full( (num_images, height, width), np.nan, dtype=np.float32)
dates = []
for idx, file in enumerate(f_names):
    dates.append(os.path.basename(file)[-14:-4])
    
    ### Removing -9999.0
    img = read_pic(file)
    img = np.where(img == -9999.0, np.nan, img)
    image_stack[idx] = img
    #image_stack[idx] = read_pic(file)
    
    
    if idx % 50 == 0:
        print("ind:", idx, f"of {len(f_names)}")

mask = ~np.isnan(image_stack)
mask_any = np.any(mask, axis=0).astype(np.uint8)

print("End reading data\ninitiating smoothing\n")

# Smoothing
#smoothed_stack = gaussian_filter(image_stack, sigma=(sigma_t, sigma_x, sigma_y))
smoothed_stack = nan_gaussian_filter3d(image_stack, sigma=(sigma_t, sigma_x, sigma_y))
for i in range(smoothed_stack.shape[0]):
    smoothed_stack[i][mask_any == 0] = np.nan

smoothed_stack = np.where(np.isnan(smoothed_stack), -9999, smoothed_stack)
meta.update({"nodata": -9999.0})

print("End smoothing\nstarting saving images\n")

# Saving as images
for ind, file in enumerate(f_names):
    if (ind % 50) == 0 : print(f"run: {ind}")
    output_filename = os.path.join(out_path, f"{ext}_{dates[ind]}.tif") ## FIX IT
    save_pic(output_filename, smoothed_stack[ind])

print("End")












