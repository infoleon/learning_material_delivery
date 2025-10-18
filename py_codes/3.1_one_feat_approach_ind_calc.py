# -*- coding: utf-8 -*-


"""
One feature approach (P90 NDWI) to correct the coarse prediction.

steps
calculate the VI from one image only


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




import os
import pandas as pd
import numpy as np

import rasterio
from rasterio.transform import rowcol

from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import LeaveOneOut



import sys

import matplotlib
matplotlib.use("TkAgg")



##########################
### Training the Model ###
##########################


### path to the true soil data
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, ".." )
file_true_soil = os.path.join(  base_path , "data"  , "true_data" ,  "selection_GIS_32633_ws.csv" )


### path to the soil image

path_one_imag = os.path.join(  base_path , r"data\all_imags_raw")

file_one_imag = os.path.join(   path_one_imag , r"S2_2020-09-20.tif")
file_one_imag_filt = os.path.join(path_one_imag , r"S2_2020-09-20_SCL.tif")


### path to output
#path_out = r"D:\_Trabalho\_Publicacoes_nossas\Learning_Material_Workshop\material_Boo\coarse_map" + os.sep
path_out = os.path.join(script_dir, "..", "coarse_map")
save_images = False
save_images = True


#out_merged_csv = r"D:\_Trabalho\_Publicacoes_nossas\Learning_Material_Workshop\material_Boo\training_12_pts\monica_files\selection_GIS\_pred_vs_ref_linear.csv"
out_merged_csv = os.path.join( base_path , "data/true_data/_pred_vs_ref_linear.csv")
save_merged_csv = False
save_merged_csv = True



### Function definitions
def read_pic(tif_file):
    """ Reads a TIFF file using rasterio """
    with rasterio.open(tif_file) as f:
        bands = f.read()
        meta = f.meta  # Save metadata for output
    return bands, meta
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
def remove_bad_pixels(bands_imags, filter_imag, class_vector):
    """ Filters out unwanted pixels using classification image """
    imags = bands_imags
    output_imags = np.zeros(imags.shape)
    
    mask = np.isin(filter_imag, list(class_vector))
    output_imags = np.where(mask, np.nan, imags)
    return output_imags

### reading and processing the TRUE data (13 pts)

true_data_all = pd.read_csv(file_true_soil)


### reading and processing the bare soil image
data_imag, meta = read_pic(  file_one_imag  )

mask_imag = ~np.isnan(data_imag[0])


remove_sce_labels = [1, 2, 3, 8, 9, 10]
filter_imag, _ = read_pic(   file_one_imag_filt  )

filtered_imag = remove_bad_pixels(data_imag, filter_imag, remove_sce_labels)

bsi_data   = calculate_bsi(     filtered_imag[10], filtered_imag[3], filtered_imag[7], filtered_imag[1])
ndwi_data  = calculate_ndwi(    filtered_imag[7] , filtered_imag[10]  )



#######################
### Selected images ###
#######################

path_one_imag_bsi = os.path.join(  base_path , r"data\interpolated_images\_gauss_filt\bsi\bsi_2020-09-20.tif")
path_one_imag_ndwi = os.path.join(  base_path , r"data\interpolated_images\_gauss_filt\ndwi\ndwi_2020-09-20.tif")


bsi_data, meta   = read_pic(  path_one_imag_bsi   )
ndwi_data, meta  = read_pic(  path_one_imag_ndwi  )

bsi_data  = bsi_data[0]
ndwi_data = ndwi_data[0]



### Selecting the image data ###
# transforming the coordinates to indexes
top = true_data_all[true_data_all["Horizon_number"] == 1].copy()
xs = top[["Easting"]].to_numpy(dtype=float)
ys = top[["Northing"]].to_numpy(dtype=float)


H, W = bsi_data.shape
rows, cols = rowcol(meta["transform"], xs, ys)
rows = np.asarray(rows, dtype=int)
cols = np.asarray(cols, dtype=int)

bsi_pts  = bsi_data[rows, cols]
ndwi_pts = ndwi_data[rows, cols]


x = np.c_[
    (bsi_pts),
    (ndwi_pts),
    ]


y_soc  = top["SOC"].to_numpy(float)
y_sand = top["SAND"].to_numpy(float)
y_clay = top["CLAY"].to_numpy(float)

### Analysis of the data
print("--- Raw data analysis ---")
print(f"SOC  max: {max(y_soc):06.3f}  min: {min(y_soc):06.3f}  mean: {np.mean(y_soc):.3f}  std: {np.std(y_soc):.3f}")
print(f"Sand max: {max(y_sand):06.3f}  min: {min(y_sand):06.3f}  mean: {np.mean(y_sand):.3f}  std: {np.std(y_sand):.3f}")
print(f"Clay max: {max(y_clay):06.3f}  min: {min(y_clay):06.3f}  mean: {np.mean(y_clay):.3f}  std: {np.std(y_clay):.3f}")

### Fitting the model with linear model
def loo_linear(X, y):
    loo = LeaveOneOut()
    yhat = np.empty_like(y, dtype=float)
    for tr, te in loo.split(X):
        m = LinearRegression()
        m.fit(X[tr], y[tr])
        yhat[te] = m.predict(X[te])
    rmse = float(np.sqrt(np.mean((yhat - y)**2)))
    r2   = 1 - float(((yhat - y)**2).sum()/((y - y.mean())**2).sum())
    final = LinearRegression().fit(X, y)   # refit on all data for mapping
    return final, yhat, rmse, r2

model_soc,  yhat_soc,  rmse_soc,  r2_soc  = loo_linear(x, y_soc )
model_sand, yhat_sand, rmse_sand, r2_sand = loo_linear(x, y_sand)
model_clay, yhat_clay, rmse_clay, r2_clay = loo_linear(x, y_clay)

print("\n--- LOO analysis ---")
print(f"SOC  (top)  LOO RMSE={rmse_soc:.3f},  R²={r2_soc:.3f}, n={len(y_soc)}, cv={rmse_soc/np.mean(yhat_soc):.3f}")
print(f"SAND (top)  LOO RMSE={rmse_sand:.3f},  R²={r2_sand:.3f}, n={len(y_sand)}, cv={rmse_sand/np.mean(yhat_sand):.3f}")
print(f"CLAY (top)  LOO RMSE={rmse_clay:.3f},  R²={r2_clay:.3f}, n={len(y_clay)}, cv={rmse_clay/np.mean(yhat_clay):.3f}")
print()




def train_metrics(model, x, y, name=""):
    yhat = model.predict(x)
    rmse = float(np.sqrt(np.mean((yhat - y)**2)))
    r2   = 1 - float(((yhat - y)**2).sum() / ((y - y.mean())**2).sum())
    rmse_null = float(np.sqrt(np.mean((y - y.mean())**2)))  # baseline = predict mean
    print(f"{name}  TRAIN RMSE={rmse:.3f},  R²={r2:.3f},  RMSE/Null={rmse/rmse_null:.2f}")
    return yhat, rmse, r2

# X, y_soc, y_sand, y_clay are your design/targets after masking
_ = train_metrics(model_soc,  x, y_soc,  "SOC  (top)")
_ = train_metrics(model_sand, x, y_sand, "SAND (top)")
_ = train_metrics(model_clay, x, y_clay, "CLAY (top)")

### Generating the whole maps



h, w = bsi_data.shape
valid = np.isfinite(bsi_data)
valid_ndwi = np.isfinite(ndwi_data)

if not np.array_equal(valid, valid_ndwi):
    raise ValueError("Different image sizes from NDWI and BSI")

x_all = np.c_[
    bsi_data[valid],
    (ndwi_data[valid_ndwi]),
    ]

soc_top_all  = model_soc.predict(x_all)
sand_top_all = model_sand.predict(x_all)
clay_top_all = model_clay.predict(x_all)


soc_img  = np.full((h, w), np.nan, dtype=np.float32)
soc_img [valid] = soc_top_all
sand_img = np.full((h, w), np.nan, dtype=np.float32)
sand_img[valid] = sand_top_all
clay_img = np.full((h, w), np.nan, dtype=np.float32)
clay_img[valid] = clay_top_all



# SOC in % (adjust bounds to your units)
soc_img  = np.clip(soc_img,  0.2, 3.1)


# texture closure 0–100
sand_img = np.clip(sand_img, 45.0, 95.0)
clay_img = np.clip(clay_img, 1.0, 10.0)
silt_img = 100.0 - sand_img - clay_img
#silt_img = np.clip(silt_img, 1.0, 99.0)


def save_float32_tif(arr, meta, out_path, nodata_val=np.nan):
    m = meta.copy(); m.update(count=1, dtype="float32", nodata=nodata_val)
    to_write = np.where(np.isfinite(arr), arr.astype(np.float32), nodata_val)
    with rasterio.open(out_path, "w", **m) as dst:
        dst.write(to_write, 1)


soc_img  = np.where(mask_imag, soc_img, np.nan)
sand_img = np.where(mask_imag, sand_img, np.nan)
clay_img = np.where(mask_imag, clay_img, np.nan)

if save_images:
    save_float32_tif(soc_img,  meta, os.path.join( path_out , r"SOC_coarse_top.tif" )    )
    save_float32_tif(sand_img, meta, os.path.join( path_out , r"SAND_coarse_top.tif")    )
    save_float32_tif(clay_img, meta, os.path.join( path_out , r"CLAY_coarse_top.tif")    )



# read reference points


## true soil data all
path_true_soil_all = os.path.join( base_path , "data/true_data")
file_true_soil = os.path.join( path_true_soil_all ,  r"soil_Boo_lean_all_top.csv")

data_all_geoph = pd.read_csv(file_true_soil)
xcol, ycol = "Easting", "Northing"

xs = data_all_geoph[xcol].to_numpy(float)
ys = data_all_geoph[ycol].to_numpy(float)

# convert to (row, col) on this grid
H, W = soc_img.shape
rows, cols = rowcol(meta["transform"], xs, ys)
rows = np.asarray(rows, dtype=int)
cols = np.asarray(cols, dtype=int)

inside = (rows>=0)&(rows<H)&(cols>=0)&(cols<W)
data_all_geoph = data_all_geoph.loc[inside].copy()
rows, cols = rows[inside], cols[inside]

# sample predictions
soc_pred  = soc_img [rows, cols]
sand_pred = sand_img[rows, cols]
clay_pred = clay_img[rows, cols]

# attach indices and predictions
data_all_geoph["row"] = rows
data_all_geoph["col"] = cols
data_all_geoph["SOC_pred"]  = soc_pred
data_all_geoph["SAND_pred"] = sand_pred
data_all_geoph["CLAY_pred"] = clay_pred

# write merged CSV
if save_merged_csv:
    data_all_geoph.to_csv(out_merged_csv, index=False)
    #print("Wrote:", out_csv, "rows:", len(data_all_geoph))



### PLOT ###
#plt.figure(figsize=(5,5))
#plt.scatter(y_soc, yhat_soc, s=40)
#lo, hi = float(min(y_soc.min(), yhat_soc.min())), float(max(y_soc.max(), yhat_soc.max()))
#plt.plot([lo,hi],[lo,hi],'--'); plt.xlabel("Observed SOC"); plt.ylabel("LOO-predicted SOC"); plt.grid(True); plt.tight_layout(); plt.show()

#plt.figure(figsize=(5,5))
#plt.scatter(y_sand, yhat_sand, s=40)
#lo, hi = float(min(y_sand.min(), yhat_sand.min())), float(max(y_sand.max(), yhat_sand.max()))
#plt.plot([lo,hi],[lo,hi],'--'); plt.xlabel("Observed SOC"); plt.ylabel("LOO-predicted SOC"); plt.grid(True); plt.tight_layout(); plt.show()

#plt.figure(figsize=(5,5))
#plt.scatter(y_clay, yhat_clay, s=40)
#lo, hi = float(min(y_clay.min(), yhat_clay.min())), float(max(y_clay.max(), yhat_clay.max()))
#plt.plot([lo,hi],[lo,hi],'--'); plt.xlabel("Observed SOC"); plt.ylabel("LOO-predicted SOC"); plt.grid(True); plt.tight_layout(); plt.show()

sys.exit()











