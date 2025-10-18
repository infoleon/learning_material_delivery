# -*- coding: utf-8 -*-


"""

Generate SOC coarse map

"""


import os
import sys
import pandas as pd
import numpy as np

import rasterio
from rasterio.transform import rowcol

from sklearn.linear_model import RidgeCV, LinearRegression
from scipy.optimize import least_squares
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


import rasterio
from rasterio.transform import rowcol



path = r"../data/true_data" + os.sep

file_in = path + "selection_GIS_32633_ws.csv"
file_out = path + "map_soc.csv"


### Output path
path_coarse_map = r"../coarse_map" + os.sep
coarse_map_f = path_coarse_map + "SOC_coarse_top.tif"


cols = ["SAND", "SILT", "CLAY", "KA5", "SOC", "BD"]

data_in = pd.read_csv(file_in)


def loo_linear(X, y):
    loo = LeaveOneOut()
    yhat = np.empty_like(y, dtype=float)
    for tr, te in loo.split(X):
        m = LinearRegression()
        m.fit(X[tr], y[tr])
        yhat[te] = m.predict(X[te])
    rmse = float(np.sqrt(np.mean((yhat - y)**2)))
    r2 = 1 - float(((yhat - y)**2).sum()/((y - y.mean())**2).sum())
    final = LinearRegression().fit(X, y)   # refit on all data for mapping
    return final, yhat, rmse, r2



data_in = data_in.drop(data_in[data_in["Horizon_number"] == 4].index)

soc0_t = data_in[data_in["Horizon_number"] == 1]
soc0 = soc0_t["SOC"]
soc0 = np.ravel([soc0, soc0, soc0], 'F')  # soc0 repeated 3 times


z = (data_in["Horizon_number"]-1).to_numpy(float)
yy = data_in["SOC"].to_numpy(dtype=float)


x = np.c_[
    soc0 * 1/np.exp(z),
    ]

y = np.c_[
    yy
    ]

model_soc_2,  yhat_soc_2,  rmse_soc_2,  r2_soc_2 = loo_linear(x, y)


print("\n--- LOO analysis ---")
print(f"SOC (2nd)  LOO RMSE={rmse_soc_2:.3f},  R²={r2_soc_2:.3f}, n={len(y)}, cv={rmse_soc_2/np.mean(yhat_soc_2):.3f}")


##########################
#### Make soc 2nd lay ####
##########################

print("Starting production of soc in the second and third layer")



# helper functions
def read_pic(tif_file):
    """ Reads a TIFF file using rasterio """
    with rasterio.open(tif_file) as f:
        bands = f.read()
        meta = f.meta  # Save metadata for output
    return bands, meta

def predict_layer(model, soc0_map, z):
    feat = soc0_map * np.exp(-float(z))              # x = SOC0 * e^{-z}
    out = np.full_like(feat, np.nan, dtype=np.float32)
    m = np.isfinite(feat)
    if m.any():
        out[m] = model.predict(feat[m].reshape(-1, 1)).ravel().astype(np.float32)
    return out

def save_float32_tif(arr, meta, out_path, nodata_val=np.nan):
    m = meta.copy()
    m.update(count=1, dtype="float32", nodata=nodata_val)
    to_write = np.where(np.isfinite(arr), arr.astype(np.float32), nodata_val)
    with rasterio.open(out_path, "w", **m) as dst:
        dst.write(to_write, 1)

def clip_array(arr, vmin=None, vmax=None):
    """Clip array to [vmin, vmax], ignoring NaNs."""
    out = arr.copy()
    if vmin is not None:
        out = np.where(np.isfinite(out), np.maximum(out, vmin), out)
    if vmax is not None:
        out = np.where(np.isfinite(out), np.minimum(out, vmax), out)
    return out


top_lay, meta = read_pic(coarse_map_f)
top_lay = top_lay[0].astype(np.float64)

# predict z=1 (2nd layer) and z=2 (3rd layer)
soc_z1 = predict_layer(model_soc_2, top_lay, z=1)
soc_z2 = predict_layer(model_soc_2, top_lay, z=2)


soc_z1 = np.clip(soc_z1, 0.01, 1.7)
soc_z2 = np.clip(soc_z2, 0.01, 0.6)

save_float32_tif(soc_z1, meta, os.path.join(path_coarse_map, "SOC_coarse_layer2.tif"))
save_float32_tif(soc_z2, meta, os.path.join(path_coarse_map, "SOC_coarse_layer3.tif"))




from joblib import dump
models_dir = os.path.join("..", "models")
os.makedirs(models_dir, exist_ok=True)
dump(model_soc_2, os.path.join(models_dir, "model_soc_2.joblib"))



































