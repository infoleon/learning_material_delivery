


# -*- coding: utf-8 -*-



import os
import sys

import pandas as pd
from copy import deepcopy

import json
import shutil


from datetime import datetime, timedelta



### Read template files and stores it

class read_mon_files:
    def __init__(self, soil_file, template_sim, template_site, template_crop, template_weather, dens, field, weather_csv_separator, header_weather):
        
        # START input data #####
        self.soil_path_file = soil_file
        
        self.field = field
        self.dens = dens
        ### END input data #####
        
        # Weather file info
        self.weather = pd.read_csv(template_weather, encoding='unicode_escape', sep = weather_csv_separator, header = header_weather)
        # read Templates
        with open(template_sim ,encoding='utf-8') as f:
            self.sim_path_d   = json.load(f)
        with open(template_site,encoding='utf-8') as f:
            self.site_data    = json.load(f)
        with open(template_crop,encoding='utf-8') as f:
            self.crop_path_d  = json.load(f)
        
        # use pandas to read CSV raw data file
        self.df = pd.read_csv(self.soil_path_file)  # read soil data
        #filter the data to the selected parameters
        #self.df_filter = self.df[(self.df["Field_name"] == self.field) & (self.df["compaction_level"] == self.dens)]
        
        self.df_filter = deepcopy(self.df)
        self.df_filter.loc[:, "Thickness"] = self.df_filter["Lower_depth"] - self.df_filter["Upper_depth"]
        # sorting based on point then layer
        self.df_filter = self.df_filter.sort_values(by=['Field_Profile_number', 'Horizon_number'])
        
        # filtering data
        self.lay_thic = self.df_filter["Thickness"] / 100  # /100 to transform to meters
        self.soc      = self.df_filter["SOC"]
        self.BD       = self.df_filter["BD"] * 1000.0  # from g/cm3 to kg/m3
        
        # texture
        self.tex  = self.df_filter["KA5"]
        self.sand = self.df_filter["SAND"] / 100.0  # transform from 0 to 1
        self.clay = self.df_filter["CLAY"] / 100.0
        
        # get the field point IDs
        self.point_ID = self.df_filter["Field_Profile_number"]
        self.val_uniq = self.point_ID.unique()
        # self.val_uniq carries the number of simulations per batch
        
        # meta
        self.weath_files   = []
        self.gen_sim_dat   = []
        self.data_crop     = []  # list containing all the crop files as json format
        self.point_ID_log  = []
        self.coord = []
        self.coor_X = self.df_filter["Easting"]
        self.coor_Y = self.df_filter["Northing"]
    #generate the soil files and return a list containing all of the json_files
    
    
    
    def gen_sites(self):
        # create dictionaries, for point and layers
        # dictinoary and list
        self.point_ID_log = []
        self.coord = []
        self.data_field_soil = []    # soil data for each point, list with json formats readily available to send to MONICA
        
        count_ini, count_end = int(0), int(0)
        
        #print(self.val_uniq)
        
        # points in a field loop
        for i in range(len(self.val_uniq)):
            count_ini += count_end
            count_end = int( (self.point_ID == self.val_uniq[i]).sum() )
            
            lis_lay=[]
            self.point_ID_log.append(self.point_ID.iloc[count_ini])
            self.coord.append((self.coor_X.iloc[count_ini], self.coor_Y.iloc[count_ini]))
            
            # layers in a point loop
            for ii in range(count_ini, count_ini+count_end, int(1)):
                th_  = round(float(self.lay_thic.iloc[ii]) , 6)  
                soc_ = round(float(self.soc.iloc[ii])      , 6)
                k5_  = self.tex.iloc[ii]
                sand_ = round(float(self.sand.iloc[ii]) , 3)
                clay_ = round(float(self.clay.iloc[ii]) , 3)
                bd_  = round(float(self.BD.iloc[ii])    , 6)
                
                dc_  = {
                    "Thickness": [th_, "m"],
                    "SoilOrganicCarbon": [soc_, "%"],
                    "KA5TextureClass": k5_,
                    "Sand": sand_,
                    "Clay": clay_,
                    "SoilBulkDensity": [bd_, "kg m-3"]}
                lis_lay.append(dc_)
            
            # copy Json file
            js_file = deepcopy(self.site_data)
            
            # Alter json file
            js_file["SiteParameters"]["SoilProfileParameters"] = lis_lay
            self.data_field_soil.append(js_file)
            
        return self.data_field_soil
    
    
    def gen_crop(self):
        self.data_crop=[]
        js_f_t = deepcopy(self.crop_path_d)
        for i in range(len(self.val_uniq)):
            #### CHANGE CROP FILE HERE IF NECESSARY ####
            self.data_crop.append(deepcopy(js_f_t))
        return self.data_crop
    
    
    def gen_weather(self):
        
        f_t = deepcopy(self.weather)
        original_header = f_t.columns
        
        # Remove headers
        f_t.columns = range(f_t.shape[1])
        
        f_t.iloc[:, 0] = pd.to_datetime(f_t.iloc[:, 0], format='%Y-%m-%d', errors='coerce')
        
        start_date  = datetime.strptime(self.sim_path_d["climate.csv-options"]["start-date"]   ,  "%Y-%m-%d")#.date().isoformat()
        end_date    = datetime.strptime(self.sim_path_d["climate.csv-options"]["end-date"]     ,  "%Y-%m-%d")#.date().isoformat()
        
        f_t = f_t[(f_t.iloc[:, 0] >= start_date) & (f_t.iloc[:, 0] <= end_date)]
        f_t.iloc[:, 0] = f_t.iloc[:, 0].apply(lambda x: x.strftime('%Y-%m-%d'))
        
        f_t.columns = original_header
        
        self.weath_files = []
        for i in range(len(self.val_uniq)):
            #### CHANGE WEATHER FILE HERE ####
            self.weath_files.append(deepcopy(f_t))
        return self.weath_files
    
    
    # generate sim file and organizer file
    def gen_sim(self):
        self.gen_sim_dat = []
        temp_sim = deepcopy(self.sim_path_d)
        # points in a field loop
        for i in range(len(self.val_uniq)):
            self.gen_sim_dat.append(temp_sim)
        
        return self.gen_sim_dat


# Test
if __name__ == "__main__":
    
    raw_data_path     = r"D:\_Trabalho\_Publicacoes_nossas\Inv_Mod_SHP\__MONICA_New\_Boo_Monica\input_files\raw_data" + os.sep
    soil_file         = raw_data_path + "soil_Boo_lean.csv"
    template_sim      = raw_data_path + r"sim_template.json"
    template_site     = raw_data_path + r"site_template.json"
    template_crop     = raw_data_path + r"crop_template.json"
    template_weather  = raw_data_path + "_template_weather.csv"
    
    dens = "middle"
    field = "1211"
    csv_separator = "\t"
    
    #path_all_out = r"D:\_Trabalho\_Publicacoes_nossas\Inv_Mod_SHP\__MONICA_New\projects\Work_single_run\soils_fit" + os.sep
    
    soil_files = read_mon_files(soil_file, template_sim, template_site, template_crop, template_weather, dens, field, csv_separator)
    sites_j    = soil_files.gen_sites()
    crops_j    = soil_files.gen_crop()
    weathers_j = soil_files.gen_weather()
    sims_j     = soil_files.gen_sim()














