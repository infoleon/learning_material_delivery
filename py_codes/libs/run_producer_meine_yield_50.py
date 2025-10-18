#!/usr/bin/python
# -*- coding: UTF-8

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */

# Authors:
# Michael Berg-Mohnicke <michael.berg@zalf.de>
#
# Maintainers:
# Currently maintained by the authors.
#
# This file has been created at the Institute of
# Landscape Systems Analysis at the ZALF.
# Copyright (C: Leibniz Centre for Agricultural Landscape Research (ZALF)



import json
import sys
import zmq
import os
from . import monica_io3
import errno

from datetime import datetime, timedelta
import time

import pandas as pd
import sys



#print(sys.path)

#print("pyzmq version: ", zmq.pyzmq_version(), " zmq version: ", zmq.zmq_version())

# just to know which index is which
# CLIMATE_VARIABLES = {
#     '3':  ['tasmin'  ,  2] ,  # temp min
#     '4':  ['tas'     ,  1] ,  # temp media
#     '5':  ['tasmax'  ,  3] ,  # temp max
#     '6':  ['pr'      ,  6] ,  # chuva
#     '8':  ['rsds'    ,  5] ,  # radiacao
#     '9':  ['sfcWind' ,  4] ,  # vento
#     '12': ['hurs'    ,  7] ,  # humidade
#     }



def set_nested_value(target_dict, keys, value):
    d = target_dict
    # Convert string indices to integers for list indexing
    for key in keys[:-1]:
        # Convert the key to an integer if it looks like one
        if key.isdigit():
            key = int(key)
        d = d[key]
    
    # Final assignment
    final_key = keys[-1]
    if final_key.isdigit():
        final_key = int(final_key)
    d[final_key] = value



def run_producer_50(sim_fs, site_fs, crop_fs, weather_fs, path_dir, path_env, weather_header_order,
                 new_env_value , len_points, outros):
    
    context = zmq.Context()
    socket = context.socket(zmq.PUSH) # pylint: disable=no-member
    
    server = {"server": None, "port": None}
    
    #os.chdir(r"D:\_Trabalho\_Publicacoes_nossas\Inv_Mod_SHP\__MONICA_New\code_call_Mon\_MONICA_RAW\monica_win64_3.6.12\projects\Hohenfinow2-producer-consumer")
    #os.environ["MONICA_PARAMETERS"] = r"D:\_Trabalho\_Publicacoes_nossas\Inv_Mod_SHP\__MONICA_New\code_call_Mon\_MONICA_RAW\monica_win64_3.6.12\projects\Hohenfinow2-producer-consumer\monica-parameters"
    
    os.chdir(path_dir)
    os.environ["MONICA_PARAMETERS"] = path_env
    
    
    
    config = {
        "port": server["port"] if server["port"] else "6666",
        #"server": server["server"] if server["server"] else "login01.cluster.zalf.de",#"localhost",
        "server": server["server"] if server["server"] else "localhost",
        #"sim.json":  os.path.join(os.path.dirname(__file__), '../sim-min.json'),
        #"crop.json": os.path.join(os.path.dirname(__file__), '../crop-min.json'),
        #"site.json": os.path.join(os.path.dirname(__file__), '../site-min.json'),
        #"climate.csv": os.path.abspath(os.path.join(os.path.dirname(__file__), '../climate-min.csv')),
        "debugout": "debug_out",
        "writenv": True,
        "shared_id": None,
        }
    
    
    # read commandline args only if script is invoked directly from commandline
    if len(sys.argv) > 1 and __name__ == "__main__":
        for arg in sys.argv[1:]:
            k, v = arg.split("=")
            if k in config:
                if k == "writenv" :
                    config[k] = bool(v)
                else :
                    config[k] = v
                    
    #print("config:", config)
    
    
    
    #print("conecting to Monica Server...")
    socket.connect("tcp://" + config["server"] + ":" + config["port"])
    
    #static_wet = r"D:\_Trabalho\_Publicacoes_nossas\rain_fit_10_pts_09_24\fit_plant_monica_Boo\data\_template_weather.csv"
    
    #########################################################
    ##### --- Change here for different fittings!!! --- #####
    #########################################################
    
    
    # Important to create the env outside the main loop
    sim_json  = sim_fs[0]
    site_json = site_fs[0]
    crop_json = crop_fs[0]
    wet_data  = weather_fs[0]
    env = monica_io3.create_env_json_from_json_config({
        "crop": crop_json,
        "site": site_json,
        "sim" : sim_json,
        "climate": "" #climate_csv
        })
    
    
    ################################
    ##### WEATHER DATA PROCESS #####
    ################################
    
    start_date  = datetime.strptime(sim_json["climate.csv-options"]["start-date"]   ,  "%Y-%m-%d").date().isoformat()
    end_date    = datetime.strptime(sim_json["climate.csv-options"]["end-date"]     ,  "%Y-%m-%d").date().isoformat()
    #t1 = time.perf_counter()
    #print("prod: i:", i, " t1-s:", t1 - start_time, "seconds")
    
    _ = weather_header_order
    climate_data = {
        '3':  wet_data.iloc[:,_[0]].values.tolist(),
        '4':  wet_data.iloc[:,_[1]].values.tolist(),
        '5':  wet_data.iloc[:,_[2]].values.tolist(),
        '6':  wet_data.iloc[:,_[3]].values.tolist(),
        '8':  wet_data.iloc[:,_[4]].values.tolist(),
        '9':  wet_data.iloc[:,_[5]].values.tolist(),
        '12': wet_data.iloc[:,_[6]].values.tolist(),
        }
    
    #t2 = time.perf_counter()
    #print("prod: i:", i, " t2-t1:", t2 - t1, "seconds")
    
    
    climate_json = {
        "type": "DataAccessor",
        "data": climate_data,
        "startDate": start_date,
        "endDate": end_date,
      }
    
    env["climateData"] = climate_json # HERE ALL DATA AS JSON FILE
    
    ################################
    ##### END END END END END ######
    ################################
    
    
    #print("initiating loop...")
    leng = len(sim_fs)
    for i in range(leng):
        #start_time = time.perf_counter()
        
        #if ((i + 1) % 1000) == 0:
        #    print("run producer:", i + 1)
        
        
        ##########################################
        ##### --- CHANGE PARAMETERS HERE --- #####
        ##########################################
        
        
        #sim_json  = sim_fs[i]
        site_json = site_fs[i]["SiteParameters"]
        #crop_json = crop_fs[i]
        #wet_data  = weather_fs[i]
        #wet_data  = weather_fs
        
        
        # Changing soils
        env["params"]["siteParameters"] = site_json
        
        # Crop changes
        env["cropRotation"][0]["worksteps"][1]["crop"]["cropParams"]["cultivar"]["StageTemperatureSum"][0] = new_env_value
        env["cropRotation"][0]["worksteps"][1]["crop"]["cropParams"]["cultivar"]["VernalisationRequirement"][0] = [0,outros[0],0,0,0,0]
        env["cropRotation"][0]["worksteps"][1]["crop"]["cropParams"]["cultivar"]["MaxAssimilationRate"] = outros[1]
        
        #aa=env["cropRotation"][0]["worksteps"][1]["crop"]["cropParams"]["cultivar"]["StageTemperatureSum"][0]
        #print(aa)
        
        
        env["csvViaHeaderOptions"] = sim_json["climate.csv-options"]
        env["pathToClimateCSV"]    = ""#static_wet
        
        # send number and total
        env["customId"] = [i, leng]
        
        
        
        # add shared ID if env to be sent to routable monicas
        if config["shared_id"]:
            env["sharedId"] = config["shared_id"]
            
        #if config["writenv"] :
            #filename = os.path.join(os.path.dirname(__file__), config["debugout"], 'generated_env.json')
            #WriteEnv(filename, env)
        
        #end_time = time.perf_counter()
        #print("prod: i:", i, " e-t2:", end_time - t2, "seconds")
        #print("prod: i:", i, "e-s:", end_time - start_time, "seconds")
        #env["customId"].append(end_time)
        socket.send_json(env)
        




def WriteEnv(filename, env) :
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'w') as outfile:
        json.dump(env, outfile, indent=2)






if __name__ == "__main__":
    run_producer()

