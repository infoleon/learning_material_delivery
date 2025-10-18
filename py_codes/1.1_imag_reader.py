
# -*- coding: utf-8 -*-

"""

Generate timeseries of a pixel with the data and timeseries interpolation
Just one pixel for analysis

"""


import rasterio
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import sys
import os
import glob

from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
#from statsmodels.nonparametric.smoothers_lowess import lowess

#from pykalman import KalmanFilter

from scipy.sparse import diags
from scipy.linalg      import cho_factor, cho_solve




whittaker_ = True


lambda_whittaker = 100  # Adjust this parameter for more or less smoothing

px_x = 60
px_y = 45

ndwi    = False    # 10_000
bsi     = True    # 10_000


"""
sowing - 2020-09-17
crop   - 2021-08-23

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


# FUNCS #######################################################################
def read_pic(tif_file):
    with rasterio.open(tif_file) as f:
        bands = f.read()
    return bands

def remove_bad_pixels(bands_imags, filter_imag, class_vector):
    """
    bands_imag: raw data images (vector of images)
    filter_imag: classification image
    class_vector: classification of the filter_imag, which will be removed
    """
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


# END FUNCS ###################################################################



#all imags
work_path = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(work_path, "..")
path = os.path.join( base_path , "data/all_imags_raw" )

remove_sce_labels = [1, 2, 3, 8, 9, 10]

# file names
f_names = sorted(glob.glob(f"{path}/*.tif"))

pairs = [f_names[i:i + 2] for i in range(0, len(f_names), 2)]


filtered_data_output = []
timeseries = []
for pair in pairs:
    data_imag = read_pic(pair[0])
    filter_imag = read_pic(pair[1])
    
    filtered_imag = remove_bad_pixels(data_imag, filter_imag[0], remove_sce_labels)
    date_imag = datetime.strptime(pair[0][-14:-4], "%Y-%m-%d").date()
    
    if bsi  : indexes = calculate_bsi(           filtered_imag[10] , filtered_imag[3], filtered_imag[7], filtered_imag[1])
    #print(filtered_imag[10][10] , filtered_imag[3][10], filtered_imag[7][10], filtered_imag[1][10])
    if ndwi : indexes = calculate_ndwi(          filtered_imag[7]  , filtered_imag[10]  )
    
    filtered_data_output.append([date_imag,filtered_imag])
    timeseries.append([date_imag,   indexes   ])


values=[]
dates=[]
for i in timeseries:
    dates.append(i[0])
    values.append(i[1][px_x][px_y])



# --- Step 0: Removing nan data from the pixel ---
# All the data
valid_dates = [d for d, v in zip(dates, values) if not np.isnan(v)]
valid_ndvi = [v for v in values if not np.isnan(v)]


# Convert dates to numerical format for interpolation
dates_numeric = np.array([d.toordinal() for d in valid_dates])

valid_dates_numeric = np.array([d.toordinal() for d in valid_dates])
valid_ndvi_values = valid_ndvi


if whittaker_:
    # --- Whittaker Smoothing ---
    full_dates_numeric = np.arange(valid_dates_numeric[0], valid_dates_numeric[-1] + 1)

    # Interpolate linearly to create daily values
    linear_interp = interp1d(valid_dates_numeric, valid_ndvi_values, kind='linear', fill_value="extrapolate")
    ndvi_linear_interpolated = linear_interp(full_dates_numeric)

    # Apply smoother
    ndvi_interpolated = whittaker_smooth(ndvi_linear_interpolated, lmbd=lambda_whittaker)

    full_dates = [datetime.fromordinal(int(d)) for d in full_dates_numeric]



### PLOT
plt.figure(figsize=(12, 6))
plt.plot(valid_dates, valid_ndvi, "o", label="Original (Raw)", color="green")
plt.plot(full_dates, ndvi_interpolated, "--", label="Cubic Interpolation", color="blue")
#plt.plot(dates_smoothed, ndvi_smoothed, "-", label="LOWESS Smoothing", color="red")

plt.xlabel("Date")
plt.ylabel("Index Value")
#plt.ylim(bottom=0)
plt.title("Index Time Series")
#plt.xticks(rotation=45)
plt.grid()
plt.legend()

plt.show()

















