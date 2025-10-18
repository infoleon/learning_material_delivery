# -*- coding: utf-8 -*-


"""

Read coordinates and outputs the selected coordinates data

"""



import os
import sys

import numpy as np

import pandas as pd
import rasterio
from rasterio.transform import rowcol






# INPUTS
raster_path = "../data/interpolated_images/_gauss_filt/bsi/bsi_2020-09-20.tif"

selected_p = "../data/true_data/selected_coordinates.csv"
all_dat_p = "../data/true_data/soil_Boo_lean_all.csv"

out_file = "../data/true_data/selection_GIS_32633_ws.csv"



# Load tables
all_dat   = pd.read_csv(all_dat_p)
selec_dat = pd.read_csv(selected_p)

# Clean types
for k in ["Easting","Northing"]:
    all_dat[k]   = pd.to_numeric(all_dat[k], errors="coerce")
    selec_dat[k] = pd.to_numeric(selec_dat[k], errors="coerce")

with rasterio.open(raster_path) as src:
    print("Raster CRS:", src.crs)

    xs = selec_dat["Easting"].to_numpy()
    ys = selec_dat["Northing"].to_numpy()

    rows, cols = rasterio.transform.rowcol(src.transform, xs, ys, op=np.floor)
    
    # Filter out-of-bounds clicks
    inb = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
    rows, cols = rows[inb], cols[inb]

    # Get exact pixel centers
    centers = [src.xy(int(r), int(c), offset='center') for r, c in zip(rows, cols)]
    cx, cy = map(np.array, zip(*centers))

# Build snapped selection (dedupe multiple clicks on same pixel)
snap = pd.DataFrame({"Easting": cx, "Northing": cy}).drop_duplicates()

# Join to your data (which stores pixel-center coords)
out = all_dat.merge(snap, on=["Easting","Northing"], how="inner")
out.to_csv(out_file, index=False)

# Diagnostics
print("Selected clicks:", len(selec_dat))
print("Clicks within raster:", inb.sum())
print("Snapped unique pixels:", len(snap))
print("Rows matched:", len(out))
print("Matches per pixel (expect 4 if 4 depths):")
print(out.groupby(["Easting","Northing"]).size().value_counts().head())












































