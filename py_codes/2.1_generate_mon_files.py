# -*- coding: utf-8 -*-


"""

Generate the Monica json files from templates and previously selected points (12 points)
(require Python 3.7 +)

"""

import os
import pandas as pd
import sys
import copy
import json


script_dir = os.path.dirname(os.path.abspath(__file__))

data_path_input = os.path.join(script_dir, "..", "data", "true_data")
data_f = os.path.join(  data_path_input  ,  "selection_GIS_32633_ws.csv" )

# Open and load JSON template files
json_path_input = os.path.join( script_dir , ".." , "monica_files"  )
json_site_template = os.path.join(json_path_input , r"site_template.json")
json_sim_template  = os.path.join(json_path_input , r"sim_template.json")


with open(json_site_template, "r") as f:
    data_site_json_template = json.load(f)

with open(json_sim_template, "r") as f:
    data_sim_json_template = json.load(f)


json_path_output = os.path.join( json_path_input , "local_files_fit" )

os.makedirs(  json_path_output , exist_ok=True)


atributes = ["SAND", "SILT", "CLAY", "SOC", "BD", "KA5", "Thickness"]
layers = 4

data_soil = pd.read_csv(data_f)


# getting the IDs of the pixels
locations = data_soil["Field_Profile_number"].unique().tolist()

#print("Soil IDs:")
#for i in locations:
#    print(i)

#Grouping the data per location
grouped_soil_data = {i: group[atributes] for i, group in data_soil.groupby("Field_Profile_number")}


for location in locations:
    
    # Site json files --------------------------------------------------------------------------
    temp_site_json = copy.deepcopy(data_site_json_template)
    for layer in range(layers):
        sand = round(grouped_soil_data[location]["SAND"].iloc[layer] , 4)
        clay = round(grouped_soil_data[location]["CLAY"].iloc[layer] , 4)
        soc  = round(grouped_soil_data[location]["SOC" ].iloc[layer] , 4)
        bd   = round(grouped_soil_data[location]["BD"  ].iloc[layer] * 1000 , 4)
        ka5  = grouped_soil_data[location]["KA5" ].iloc[layer]
        thick = float(round(grouped_soil_data[location]["Thickness"].iloc[layer]/100, 4))
        
        temp_site_json["SiteParameters"]["SoilProfileParameters"][layer]['Sand'] = sand/100
        temp_site_json["SiteParameters"]["SoilProfileParameters"][layer]['Clay'] = clay/100
        temp_site_json["SiteParameters"]["SoilProfileParameters"][layer]['SoilOrganicCarbon'] = [soc, '%']
        temp_site_json["SiteParameters"]["SoilProfileParameters"][layer]['SoilBulkDensity']   = [bd, 'kg m-3']
        temp_site_json["SiteParameters"]["SoilProfileParameters"][layer]['KA5TextureClass']   =  ka5
        temp_site_json["SiteParameters"]["SoilProfileParameters"][layer]['Thickness']         = [thick , 'm']
        temp_site_json["SiteParameters"]["SoilProfileParameters"][layer].pop("KA5TextureClass", None) # remove the "KA5TextureClass" from template file
        
    # save soil json
    with open( os.path.join(json_path_output , f"site_{location}.json") , "w") as f:
        json.dump(temp_site_json, f, indent = 4)
    
    # Sim json files ---------------------------------------------------------------------------
    temp_sim_json = copy.deepcopy(data_sim_json_template)
    temp_sim_json["site.json"] = f"site_{location}.json"
    temp_sim_json["output"]["file-name"] = f"./results/out_{location}.csv"
    
    # save sim json
    with open( os.path.join(json_path_output , f"sim_{location}.json") , "w") as f:
        json.dump(temp_sim_json, f, indent = 4)



sys.exit()


