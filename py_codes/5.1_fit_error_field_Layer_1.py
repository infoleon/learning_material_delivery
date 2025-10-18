# -*- coding: utf-8 -*-

"""

Fetch data from: TRUE data, SAT maps, MONICA simulations

Train the error correction model

"""



import matplotlib.pyplot as plt

import pandas as pd
import os
import sys

import glob
import re

import rasterio as rio
from datetime import date, timedelta

import numpy as np

import rasterio
from rasterio.transform import rowcol, array_bounds

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut, cross_val_predict

from sklearn.metrics import r2_score, mean_absolute_error






"SOC_coarse_top.tif"


show_sim_maps = False
# transpiration or yield? If transpiration = True
transpiration = True
yield_sim = False
p95_images = False

### Reading the data ###

########################
### G.True data read ###
########################


#data_path_input = r"D:\_Trabalho\_Publicacoes_nossas\Learning_Material_Workshop\material_Boo\training_12_pts\monica_files\selection_GIS" + os.sep
data_path_input =  "../data/true_data" + os.sep

data_f = data_path_input + r"selection_GIS_32633_ws.csv"

data_true = pd.read_csv(data_f)
profile_id = data_true["Field_Profile_number"].unique()

### To train
data_true_soc1 =  data_true[ data_true["Horizon_number"] == 1 ]
#train_prolile_list = data_true_soc2["Field_Profile_number"].tolist()


## All True data, for checking
data_all = data_path_input + "soil_Boo_lean_all.csv"
data_all_soc2_ = pd.read_csv(data_all)

data_all_soc1 = data_all_soc2_[ data_all_soc2_["Horizon_number"] == 1 ]


########################
### Monica data read ###
########################

monica_out_path = "../monica_files/coarse_simulation/results"

monica_out = glob.glob(os.path.join(monica_out_path, "out_*"))
monica_out_sorted = sorted(monica_out, key=lambda f: int(re.search(r"out_(\d+)\.csv", os.path.basename(f)).group(1)))

meta_path = monica_out_path + os.sep
meta_f = meta_path + "_soils_metadata_32633.csv"
meta_data = pd.read_csv(meta_f)

N = len(monica_out_sorted)

def process_one(path,i,n):
    if (i % 100) == 0:
        print(f"reading out file {i} from {n}")
    i += 1
    df = pd.read_csv(path, skiprows=1, usecols=["TraDef", "Yield"])
    return {
        "file": os.path.basename(path),
        "idx": int(re.search(r"out_(\d+)\.csv", os.path.basename(path)).group(1)),
        "Tra_sum": float(df["TraDef"].sum()),
        "Yield_max": float(df["Yield"].max())
    }

rows_ = [process_one(p, i, N) for i, p in enumerate(monica_out_sorted, start=1)]
print("Finished reading")

result = pd.DataFrame(rows_).sort_values("idx").reset_index(drop=True)

keep_c = ["idx", "x", "y"]
lookup = meta_data[keep_c]
result_monica = result.merge(lookup, on="idx", how="left")


######################
### Read NDWI data ###
######################

ini_date  = "2021-02-22"
end_date  = "2021-05-17"

w_path = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.join(w_path , "..")
input_path_imag_1 = base_path + os.sep + "data/interpolated_images/_gauss_filt/ndwi" + os.sep

def read_pic(tif_file):
    with rasterio.open(tif_file) as f:
        bands = f.read()
        meta = f.meta
        meta["nodata"] = np.nan
    return bands, meta

def load_ndwi_timeseries(folder, start_date_str, end_date_str):
    sd = date.fromisoformat(start_date_str)
    ed = date.fromisoformat(end_date_str)
    files = glob.glob(os.path.join(folder, "ndwi_*.tif"))
    dated = []
    for f in files:
        m = re.match(r"ndwi_(\d{4}-\d{2}-\d{2})\.tif", os.path.basename(f))
        if m:
            d = date.fromisoformat(m.group(1))
            if sd <= d <= ed:
                dated.append((d, f))
    dated.sort(key=lambda x: x[0])
    bands, meta = read_pic(dated[0][1])
    cube = [bands[0].astype(np.float32)]
    for _, f in dated[1:]:
        b, _ = read_pic(f)
        cube.append(b[0].astype(np.float32))
    cube = np.stack(cube, axis=0)  # (T,H,W)
    dates = [d for d, _ in dated]
    return cube, dates, meta

cube, dates, meta_ndwi = load_ndwi_timeseries(input_path_imag_1, ini_date, end_date)
p95 = np.nanpercentile(cube, 95, axis=0).astype(np.float32)
p95 = np.where(p95 == -9999.0, np.nan, p95)


# --- coarse SOC raster + meta_coarse ---
path_coarse_map = "../coarse_map" + os.sep
coarse_map_f = path_coarse_map + "SOC_coarse_top.tif"
soc2_coarse_bands, meta_coarse = read_pic(coarse_map_f)
soc1_coarse = soc2_coarse_bands[0]  # (Hc,Wc)


############################
### END Reading the data ###
############################


def df_to_array_xy(df, meta, vcol, xcol="x", ycol="y", fill=np.nan):
    H, W = meta["height"], meta["width"]; T = meta["transform"]
    xs = df[xcol].to_numpy(dtype="float64")
    ys = df[ycol].to_numpy(dtype="float64")
    r, c = rowcol(T, xs, ys, op=np.floor)
    r = r.astype(np.int64); c = c.astype(np.int64)
    m = (r>=0) & (r<H) & (c>=0) & (c<W)
    arr = np.full((H, W), fill, dtype=np.float32)
    arr[r[m], c[m]] = df[vcol].to_numpy(np.float32)[m]
    return arr


# Dataframe to image (matrix)
tra_mon = df_to_array_xy(result_monica, meta_coarse, "Tra_sum")  # (Hc,Wc)
yield_max = df_to_array_xy(result_monica, meta_coarse, "Yield_max")  # (Hc,Wc)

soc_true = df_to_array_xy(
    data_all_soc1,            # df
    meta_coarse,              # same grid as soc2_coarse
    vcol="SOC",               # value column
    xcol="Easting",           # X column in your CSV
    ycol="Northing"           # Y column in your CSV
)




#if True:
if show_sim_maps:
    def plot_raster(arr, meta, title="", cmap="viridis"):
        """
        Quick plot of a 2D raster array with geo extent from metadata.
        """
        H, W = arr.shape
        west, south, east, north = array_bounds(H, W, meta["transform"])
    
        plt.figure(figsize=(6,5))
        im = plt.imshow(arr, extent=(west, east, south, north),
                        origin="upper", cmap=cmap)
        plt.colorbar(im, label=title)
        plt.title(title)
        plt.xlabel("Easting")
        plt.ylabel("Northing")
        plt.tight_layout()
        plt.show()
    
    if transpiration:
        plot_raster(tra_mon, meta_coarse, title="Tra_sum (Monica)")
    if yield_sim:
        plot_raster(yield_max, meta_coarse, title="Yield_max (Monica)")
    if p95_images:
        plot_raster(p95, meta_coarse, title="p95_NDWI")    


#########################
### Building train df ###
#########################

# --- sample coarse SOC at ground-truth points (use COARSE transform) ---
xs = data_true_soc1["Easting"].to_numpy(float)
ys = data_true_soc1["Northing"].to_numpy(float)
r_c, c_c = rowcol(meta_coarse["transform"], xs, ys, op=np.floor)
r_c = r_c.astype(int); c_c = c_c.astype(int)
Hc, Wc = soc1_coarse.shape
m_in = (r_c>=0) & (r_c<Hc) & (c_c>=0) & (c_c<Wc)


# sample features at the GT points (no need to merge on 'idx')
data_true_soc1 = data_true_soc1.copy()
data_true_soc1["SOC_coarse"] = np.nan
data_true_soc1.loc[m_in, "SOC_coarse"] = soc1_coarse[r_c[m_in], c_c[m_in]]
data_true_soc1["Tra_sum_pt"] = np.nan
data_true_soc1.loc[m_in, "Tra_sum_pt"] = tra_mon[r_c[m_in], c_c[m_in]]
data_true_soc1["p95_pt"] = np.nan
data_true_soc1.loc[m_in, "p95_pt"] = p95[r_c[m_in], c_c[m_in]]
data_true_soc1["Yield_pt"] = np.nan
data_true_soc1.loc[m_in, "Yield_pt"] = yield_max[r_c[m_in], c_c[m_in]]


data_true_soc1["SOC_true"] = np.nan
data_true_soc1.loc[m_in, "SOC_true"] = soc_true[r_c[m_in], c_c[m_in]]


train_df = (data_true_soc1.loc[m_in, ["SOC", "SOC_coarse", "Tra_sum_pt", "Yield_pt",  "p95_pt"]]   )
train_df["err"] = train_df["SOC_coarse"] - train_df["SOC"]


###############################
#### END Building train df ####
###############################


### RF on residuals ###
X = train_df[["SOC_coarse", "Tra_sum_pt", "Yield_pt" ,"p95_pt"]].to_numpy()
y = train_df["err"].to_numpy()



# --- LOOCV (out-of-fold) ---
loo = LeaveOneOut()
rf = RandomForestRegressor(n_estimators=130, max_features=int(3), min_samples_leaf=3, random_state=42, n_jobs=-1)

y_oof = cross_val_predict(rf, X, y, cv=loo, method="predict", n_jobs=-1)
r2_loo  = r2_score(y, y_oof)
mae_loo = mean_absolute_error(y, y_oof)
print(f"LOOCV R² (OOF): {r2_loo:.3f}")
print(f"LOOCV MAE (OOF): {mae_loo:.3f}")

# --- Fit on all data (raw/in-sample) ---
rf.fit(X, y)
y_pred = rf.predict(X)
r2_raw  = r2_score(y, y_pred)
mae_raw = mean_absolute_error(y, y_pred)
print(f"Raw model R²: {r2_raw:.3f}")
print(f"Raw model MAE: {mae_raw:.3f}")




######################################################################################
### Predicting for the whole map
######################################################################################

# --- predict residual over the whole grid and build CSV ---

Hc, Wc = soc1_coarse.shape
valid = np.isfinite(soc1_coarse) & np.isfinite(tra_mon) & np.isfinite(yield_max) & np.isfinite(p95) & np.isfinite(soc_true)

Xgrid = np.column_stack([
    soc1_coarse[valid],
    tra_mon[valid],
    yield_max[valid],
    p95[valid],
])
pred_err = np.full((Hc, Wc), np.nan, np.float32)
pred_err[valid] = rf.predict(Xgrid).astype(np.float32)

soc_corrected = np.where(valid, soc1_coarse - pred_err, np.nan)




### Sublayers calculations

# --- minimal helpers (only if you don't already have them) ---
def predict_layer(model, soc0_map, z):
    feat = soc0_map * np.exp(-float(z))           # x = SOC0 * e^{-z}
    out = np.full_like(soc0_map, np.nan, dtype=np.float32)
    m = np.isfinite(feat)
    if m.any():
        out[m] = model.predict(feat[m].reshape(-1,1)).ravel().astype(np.float32)
    return out

def save_float32_tif(arr, meta, path, nodata=np.nan):
    m = meta.copy(); m.update(count=1, dtype="float32", nodata=nodata)
    w = np.where(np.isfinite(arr), arr.astype(np.float32), nodata)
    with rasterio.open(path, "w", **m) as dst:
        dst.write(w, 1)


## Importing SOC saved model

from joblib import load
model_soc_2 = load(os.path.join("..", "models", "model_soc_2.joblib"))

# predict sublayers from the corrected top layer
soc_layer2_from_corr = predict_layer(model_soc_2, soc_corrected, z=1)
soc_layer3_from_corr = predict_layer(model_soc_2, soc_corrected, z=2)

# optional clipping (match what you used before)
soc_layer2_from_corr = np.clip(soc_layer2_from_corr, 0.01, 1.7)
soc_layer3_from_corr = np.clip(soc_layer3_from_corr, 0.01, 0.6)

# save
out_dir = os.path.join(base_path, "outputs_fine")
os.makedirs(out_dir, exist_ok=True)
save_float32_tif(soc_layer2_from_corr, meta_coarse, os.path.join(out_dir, "SOC_layer2_from_corrected_top.tif"))
save_float32_tif(soc_layer3_from_corr, meta_coarse, os.path.join(out_dir, "SOC_layer3_from_corrected_top.tif"))





### Here, the csv should have the "new" 2nd layer SOC data, so far it have only the first layer

# flatten to CSV with coords
rows, cols = np.where(valid)
xs, ys = rasterio.transform.xy(meta_coarse["transform"], rows, cols, offset='center')


data_all_soc2 = data_all_soc2_[data_all_soc2_["Horizon_number"] == 2]
soc_true_l2 = df_to_array_xy(data_all_soc2, meta_coarse, "SOC", xcol="Easting", ycol="Northing")


df_out = pd.DataFrame({
    "Easting":  np.asarray(xs),
    "Northing": np.asarray(ys),
    "Tra_sum":  tra_mon[valid],
    "Yield_max": yield_max[valid],
    "p95": p95[valid],
    "SOC_true_layer1": soc_true[valid],
    "SOC_coarse_layer1": soc1_coarse[valid],
    "SOC_corrected_layer1": soc_corrected[valid],
    "SOC_true_layer2": soc_true_l2[valid],
    "SOC_layer2_from_corrected": soc_layer2_from_corr[valid],
    "SOC_layer3_from_corrected": soc_layer3_from_corr[valid],
})

out_csv = os.path.join(base_path, "outputs_fine", "soc_layer1_corrected_sub_lay.csv")
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df_out.to_csv(out_csv, index=False)




















