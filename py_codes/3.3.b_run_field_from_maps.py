# -*- coding: utf-8 -*-
"""
Monica, read maps and run the whole field
"""

import os
import sys

import rasterio
from rasterio.transform import rowcol, xy

import numpy as np

import json

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor

import subprocess

import pandas as pd

from libs import bd_all_other_mineral_horizons

import copy


# Generate files? Or run?

#var = "generate"
var = "run"



if var == "generate":
    gen_files = True
    run = False
else:
    gen_files = False
    run = True


### File paths ###
path_maps = r"../coarse_map" + os.sep

f_sand = path_maps + "SAND_coarse_top.tif"
f_clay = path_maps + "CLAY_coarse_top.tif"

f_soc_1 = path_maps + "SOC_coarse_top.tif"
f_soc_2 = path_maps + "SOC_coarse_layer2.tif"
f_soc_3 = path_maps + "SOC_coarse_layer3.tif"


# MONICA parameter path
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join( script_dir, "..")

working_directory = os.path.join( base_path , r"monica_files\coarse_simulation") 
crop_folder = os.path.join(  base_path ,  r"monica_files\monica-parameters\crops") 

cultivar_sub_folder = "rye"   ### DEFINED BY USER!!!

cultivar_folder = crop_folder + os.sep + cultivar_sub_folder + os.sep
crop_file = crop_folder + os.sep + "rye_IM.json"
cultivar_file = cultivar_folder + "winter-rye_IM.json"
results_folder = working_directory + os.sep + r"results" + os.sep

os.makedirs(results_folder, exist_ok=True)

### Important!! Change to the right parameters
params = [-3.0538, 4.5808, 121.0864, 451.9509, 191.5037, 267.6732, 274.5460]

batch_size = 10

# number of layers
layers = 3
layer_thick = [0.3, 0.3, 1.4]


# Open and load JSON template files
json_path_input = base_path + os.sep + "monica_files" + os.sep
json_site_template = json_path_input + r"site_template.json"
json_sim_template  = json_path_input + r"sim_template.json"

MONICA_PARAMS = os.path.abspath(os.path.join(working_directory, "..", "monica-parameters"))
ENV = os.environ.copy()
ENV["MONICA_PARAMETERS"] = MONICA_PARAMS + os.sep


#################
### Functions ###
#################

def read_pic(tif_file):
    """ Reads a TIFF file using rasterio """
    with rasterio.open(tif_file) as f:
        bands = f.read()
        meta = f.meta  # Save metadata for output
    return bands, meta

def run_monica_for_file(file_id, working_directory, env=ENV, timeout=60000):
    res = subprocess.run(
        [MONICA_EXE, f"sim_{file_id}.json"],  # absolute exe
        cwd=working_directory,                # relative paths in JSON resolve here
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout,
    )
    if res.returncode != 0:
        raise RuntimeError(f"MONICA failed for {file_id} (rc={res.returncode}): {res.stderr.strip()}")
    return file_id

def run_monica(ids, working_directory, max_workers=6, tick=100):
    total, done, fails = len(ids), 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_monica_for_file, fid, working_directory): fid for fid in ids}
        for fut in as_completed(futs):
            fid = futs[fut]
            try:
                fut.result()
            except Exception as e:
                fails += 1
                print(f"[FAIL] {fid}: {e}", flush=True)
            finally:
                done += 1
                if done % tick == 0 or done == total:
                    print(f"[PROGRESS] {done}/{total} done ({fails} fails)", flush=True)
            
#################



# Read coarse maps
sand_m, meta_sa = read_pic(f_sand)
clay_m, meta_cl = read_pic(f_clay)

soc_m1, meta_so1 = read_pic(f_soc_1)
soc_m2, meta_so2 = read_pic(f_soc_2)
soc_m3, meta_so3 = read_pic(f_soc_3)


transform = meta_sa["transform"]
crs = meta_sa.get("crs")
H, W = sand_m.shape[1], sand_m.shape[2]
# the shape of the maps should be (1, 97, 123)

### Organizing soil data into one list
soils = []
idx = 0
for i in range(sand_m.shape[1]):
    for j in range(sand_m.shape[2]):
        sa = sand_m[0, i, j]
        cl = clay_m[0, i, j]
        s1 = soc_m1[0, i, j]
        s2 = soc_m2[0, i, j]
        s3 = soc_m3[0, i, j]
        
        si = 100 - sa - cl
        if si < 0:
            print("problem: silt value:", si)
        
        if (np.isnan(sa) or np.isnan(cl) or
                np.isnan(s1) or np.isnan(s2) or np.isnan(s3)):
            
            continue
        
        bd1 = bd_all_other_mineral_horizons(s1, sa, cl)
        bd2 = bd_all_other_mineral_horizons(s2, sa, cl)
        bd3 = bd_all_other_mineral_horizons(s3, sa, cl)
        
        x, y = xy(transform, i, j, offset='center')
        
        soils.append({
            "idx": idx,
            "row": i, "col": j,
            "x": x, "y": y,
            "sand": sa, "clay": cl,
            "soc": [s1, s2, s3],
            "bd": [bd1, bd2, bd3]
            })
        idx += 1


# Number of simulations
N = len(soils)



# Save the meta data
df = pd.DataFrame([
    {   "idx": s["idx"],
        "row": s["row"],
        "col": s["col"],
        "x": s["x"], "y": s["y"],
        "sand": s["sand"],
        "clay": s["clay"],
        "soc1": s["soc"][0],
        "soc2": s["soc"][1],
        "soc3": s["soc"][2],
        "bd1": s["bd"][0],
        "bd2": s["bd"][1],
        "bd3": s["bd"][2]  }
        for s in soils])

df.to_csv(results_folder + "_soils_metadata_32633.csv", index=False)



### Read and change crop parameters
with open(crop_file, 'r', encoding="utf-8") as f:
    crop_dat_template = json.load(f)
with open(cultivar_file, 'r', encoding="utf-8") as f:
    cultivar_dat_template = json.load(f)

x0, x1, x2, x3, x4, x5, x6 = [round(p, 6) for p in params]
#                    Base_Temp            Sum_temperature
params_unpack = [[x0,x0,  x0+2,x0+2   ,x1,x1],[x2,x3,x4,x5,x6, 20.0]]

crop_dat = crop_dat_template.copy()
cultivar_dat = cultivar_dat_template.copy()

crop_dat["BaseTemperature"] = params_unpack[0]
cultivar_dat["StageTemperatureSum"][0] = params_unpack[1]

# Save correct parameters
with open(crop_file, 'w', encoding="utf-8") as f:
    json.dump(crop_dat, f, indent=4, ensure_ascii=False)
with open(cultivar_file, 'w', encoding="utf-8") as f:
    json.dump(cultivar_dat, f, indent=4, ensure_ascii=False)



###################################
### Generate sim and soil files ###
###################################
if gen_files:
    
    # Save the meta data
    df = pd.DataFrame([
        {   "idx": s["idx"],
            "row": s["row"],
            "col": s["col"],
            "x": x, "y": y,
            "sand": s["sand"],
            "clay": s["clay"],
            "soc1": s["soc"][0],
            "soc2": s["soc"][1],
            "soc3": s["soc"][2],
            "bd1": s["bd"][0],
            "bd2": s["bd"][1],
            "bd3": s["bd"][2]  }
            for s in soils])
    
    df.to_csv(results_folder + "_soils_metadata_32633.csv", index=False)
    
    with open(json_site_template, "r") as f:
        data_site_json_template = json.load(f)
    
    with open(json_sim_template, "r") as f:
        data_sim_json_template = json.load(f)
    
    print("Generating Monica files")
    for soil in soils:
        id_ = soil["idx"]
        
        if (id_ % 100) == 0:
            print(f"run: {id_} of {N}")
        
        # Site json files --------------------------------------------------------------------------
        temp_site_json = copy.deepcopy(data_site_json_template)
        
        if "SoilProfileParameters" in temp_site_json["SiteParameters"]:
            del temp_site_json["SiteParameters"]["SoilProfileParameters"]
        temp_site_json["SiteParameters"]["SoilProfileParameters"] = [{} for _ in range(layers)]
        
        for layer in range(layers):
            sand_  = float(soil["sand"])
            clay_  = float(soil["clay"])
            soc_   = round(  float(  soil["soc"][layer] )    ,   3)
            bd_    = float( bd_all_other_mineral_horizons(soc_, sand_, clay_) )
            thick_ = float(layer_thick[layer])
            
            temp_site_json["SiteParameters"]["SoilProfileParameters"][layer]['Sand'] = sand_/100
            temp_site_json["SiteParameters"]["SoilProfileParameters"][layer]['Clay'] = clay_/100
            temp_site_json["SiteParameters"]["SoilProfileParameters"][layer]['SoilOrganicCarbon'] = [soc_, '%']
            temp_site_json["SiteParameters"]["SoilProfileParameters"][layer]['SoilBulkDensity']   = [ round(bd_ * 1000, 3), 'kg m-3']
            temp_site_json["SiteParameters"]["SoilProfileParameters"][layer]['Thickness']         = [thick_ , 'm']
            #temp_site_json["SiteParameters"]["SoilProfileParameters"][layer].pop("KA5TextureClass", None) # remove the "KA5TextureClass" from template file
        
        
        
        # save soil json
        with open( os.path.join(working_directory , f"site_{id_}.json") , "w") as f:
            json.dump(temp_site_json, f, indent = 4)
        
        # Sim json files ---------------------------------------------------------------------------
        temp_sim_json = copy.deepcopy(data_sim_json_template)
        temp_sim_json["site.json"] = f"site_{id_}.json"
        temp_sim_json["output"]["file-name"] = f"./results/out_{id_}.csv"
        
        # save sim json
        with open( os.path.join(working_directory , f"sim_{id_}.json") , "w") as f:
            json.dump(temp_sim_json, f, indent = 4)



MONICA_EXE = os.path.abspath(os.path.join(working_directory, "_monica-run.exe"))
MONICA_PARAMS_DIR = os.path.abspath(os.path.join(working_directory, "..", "monica-parameters"))
ENV = {**os.environ, "MONICA_PARAMETERS": MONICA_PARAMS_DIR}



ids = [s["idx"] for s in soils]

ids_all = [s["idx"] for s in soils]
ids = [fid for fid in ids_all if os.path.isfile(os.path.join(working_directory, f"sim_{fid}.json"))]
missing = len(ids_all) - len(ids)
if missing:
    print(f"[WARN] {missing} sim_*.json files missing; running {len(ids)} present sims.")

if run:
    run_monica(ids, working_directory, max_workers=batch_size, tick=100)















