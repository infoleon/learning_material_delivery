# -*- coding: utf-8 -*-



import pandas as pd
import geopandas as gpd
from rasterio.transform import from_origin
import rasterio
import numpy as np
import matplotlib.pyplot as plt

import os

import matplotlib.colors as mcol

from pyproj import CRS


wkt_epsg_32633 = """
PROJCS["WGS 84 / UTM zone 33N",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",15],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",0],
    AUTHORITY["EPSG","32633"],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH]]
"""


os.environ["PROJ_LIB"] = r"C:\Users\inforsato\AppData\Local\miniconda3\envs\geo\share\proj"



path_f = r"D:\_Trabalho\_Publicacoes_nossas\rain_fit_10_pts_09_24\fit_plant_monica_Boo\data_batch\results_fit_rain\qgis_plot\_results_rain.csv"




# Make a user-defined colormap.
cml = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])




df = pd.read_csv(path_f)

crs = CRS.from_wkt(wkt_epsg_32633)


gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["X"], df["Y"]), crs=crs)



# Define raster properties
x_min, y_min, x_max, y_max = gdf.total_bounds
pixel_size = 10  # Set the pixel size to your preference
width = int((x_max - x_min) / pixel_size)
height = int((y_max - y_min) / pixel_size)
transform = from_origin(x_min, y_max, pixel_size, pixel_size)

# Initialize the raster array
raster = np.zeros((height, width))



title_fig = "Diff_with_IM"
#title_fig = "Diff_no_IM"

title_fig = "rain multiplier"


#col1 = "true_val"
#col1 = "no_im"
#col1 = "with_im"#
#col1 = "dif_no_im"
#col1 = "dif_mit_im"
col1 = "rain_mult"



raster = np.full((height, width), np.nan)


# Rasterize the data using values from a specific column
for _, row in gdf.iterrows():
    x, y = row.geometry.x, row.geometry.y
    value = row[col1]  # Replace "property_column" with your actual column name
    row_idx = int((y_max - y) / pixel_size)
    col_idx = int((x - x_min) / pixel_size)
    
    if 0 <= row_idx < height and 0 <= col_idx < width:
        raster[row_idx, col_idx] = value  # Set pixel to the property value



max_value = 3_000
min_value = -4000

max_value = 1.5
min_value = 0.5

# Plot the raster
#plt.imshow(raster, extent=[x_min, x_max, y_min, y_max], cmap='viridis', vmin=min_value, vmax=max_value, interpolation='none')
#plt.imshow(raster, extent=[x_min, x_max, y_min, y_max], cmap='viridis', interpolation='none')

plt.imshow(raster, extent=[x_min, x_max, y_min, y_max], cmap=cml, vmin=min_value, vmax=max_value, interpolation='none')


plt.colorbar(label='Value Count')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(title_fig)
plt.show()



"""
with rasterio.open(
    "output_raster.tif",
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype=rasterio.float32,
    crs="EPSG:32633",
    transform=transform,
) as dst:
    dst.write(raster, 1)
"""











