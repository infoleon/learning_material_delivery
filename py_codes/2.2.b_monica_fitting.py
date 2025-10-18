# -*- coding: utf-8 -*-


"""

fitting of the selected MONICA files
BFGS Algorithm

Also compile outputs - LAI, root, stc

"""


import os
import pandas as pd
import sys

import numpy as np
import subprocess
import shutil

from scipy.optimize import minimize
from scipy.optimize import least_squares

import json

from datetime import datetime, timedelta

import time

from libs import indexes_from_coordinates
from libs import pix_timeseries
from libs import filter_vegetation_ind


from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor

import copy


"""

ML to fit using Marquardt Levenberg

"""


ML = False
#ML = True



save_output_stop = False  # Fitting, when False
save_output_stop = True


#OUT

start_t = round(time.time() , 4)

### Getting the IDs of the pixels #############
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path_input = os.path.join(  script_dir , ".." )
data_f = os.path.join( data_path_input , "data/true_data" , "selection_GIS_32633_ws.csv")


profile_id_column_name = "Field_Profile_number"

### getting the indexes from selected coordinates
ind_tag = "ndwi"
file_name_meta = rf"{ind_tag}_2019-01-14.tif"
#input_path_imag_1 = r"D:\_Trabalho\_Publicacoes_nossas\Learning_Material_Workshop\material_Boo\training_12_pts\sat_imag\interpolated_images\_gauss_filt\ndwi"  +  os.sep
#print(input_path_imag_1)

input_path_imag_1 = os.path.join( data_path_input , r"data\interpolated_images\_gauss_filt\ndwi" )


#ind_tag = "bsi"
#file_name_meta = rf"{ind_tag}_2019-01-14.tif"
#input_path_imag_1 = os.path.join( data_path_input , r"interpolated_images\_gauss_filt\bsi" )


#save_output_path = r"D:\_Trabalho\_Publicacoes_nossas\Learning_Material_Workshop\material_Boo\training_12_pts\monica_files\local_files_fit\results\veg_index_und_LAI_analis" + os.sep
save_output_path = os.path.join( data_path_input , r"monica_files\local_files_fit\results\veg_index_und_LAI_analis")
os.makedirs(  save_output_path , exist_ok=True)


### MONICA info
working_directory = os.path.join( data_path_input , r"monica_files\local_files_fit" )



# Cordinates file, the same file as the ID

### Definition and assignment of the dates for the NDVI
sowing_date  = "2021-02-22"   #!!! MOD DATE !!!!
end_date     = "2021-05-17"

# Filtering the dates by the image values
threshold = -9999


# MONICA parameter path
crop_folder = os.path.join( data_path_input , r"monica_files\monica-parameters\crops" )


cultivar_sub_folder = "rye"

crop_file =       os.path.join( crop_folder   , "rye_IM.json")
cultivar_folder = os.path.join( crop_folder  ,   cultivar_sub_folder )
cultivar_file =   os.path.join( cultivar_folder   , "winter-rye_IM.json")
results_folder =  os.path.join( working_directory , r"results"  )


## Path Parameter
path_param = os.path.join( data_path_input , "data", "fit_params" , "fit_params.json")


# Wights for "Yield", "correlation" and "peak day diff". Values multiplied by the ABS difference
weights = (1/1000, 10 , 1/100)
# Initial parameters
params_ini = [-3.0, 6.4, 154.2, 450.4, 202.0, 311.0, 301.0]

# Upper and lower bounds
bounds = [(  -5.0 ,   1.0 ),    # x0: base temperature for crop (applied to all layers)
          (   2.0 ,  11.0 ),    # x1: base temp for cultivar
          (  30.0 , 200.0 ),    # x2: stage temp sum 1  | 148
          ( 200.0 , 800.0 ),    # x3: stage temp sum 2  | 284
          ( 100.0 , 300.0 ),    # x4: stage temp sum 3  | 200
          ( 100.0 , 450.0 ),    # x5: stage temp sum 4  | 400
          ( 200.0 , 600.0 )]    # x6: stage temp sum 5  | 350



peak = False
def rmse_(val1, val2):
    return np.sqrt(np.mean(( val1 - val2) ** 2))


def run_monica_for_file(file_id, working_directory):
    os.chdir(working_directory)
    os.environ["MONICA_PARAMETERS"] = "..\\monica-parameters"
    
    #result = subprocess.run(["_monica-run.exe", f"sim_{file_id}.json"],
    #                        stdout=subprocess.PIPE,
    #                        stderr=subprocess.PIPE,
    #                        text=True)
    result = subprocess.run(["_monica-run.exe", f"sim_{file_id}.json"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    
    if result.returncode != 0:
        raise RuntimeError(f"MONICA failed for {file_id}:\n{result.stderr}")
    
    return file_id


class monica_fitting:
    def __init__(self, data_f, input_path_imag, working_directory, crop_file, cultivar_file, results_folder,
                 threshold, weights, imag_meta_data_f_name,
                 profile_id_column_name,
                 longit = "Easting", latit  = "Northing", n_layers=4, lai_vi=True):
        self.data_f = data_f
        self.input_path_imag = input_path_imag
        
        self.working_directory = working_directory
        self.crop_file = crop_file
        self.cultivar_file = cultivar_file
        self.results_folder = results_folder
        self.n_layers = n_layers
        
        self.longit = longit
        self.latit  = latit
        self.imag_meta_data_f_name = imag_meta_data_f_name
        
        self.threshold = threshold
        
        self.data_soil = pd.read_csv(data_f)
        self.locations_id = self.data_soil["Field_Profile_number"].unique().tolist()
        
        self.validate_layers()
        
        self.weights = weights
        
        self.profile_id_column_name = profile_id_column_name
        
        # dorr LAI or Tra - if false, Tra
        self.lai_vi = lai_vi
        
        ### TRUE OBS data
        self.peak_datetime = None
        self.filtered_ndvi_data = None
        self.filtered_dates = None
        self.ground_truth_yield = None
        
        self.parallel = True
        
        
        with open(self.crop_file, 'r', encoding="utf-8") as f:
            self.crop_dat_template = json.load(f)
        with open(self.cultivar_file, 'r', encoding="utf-8") as f:
            self.cultivar_dat_template = json.load(f)
    
    
    
    
    def validate_layers(self):
        if self.data_soil["Horizon_number"].max() != self.n_layers:
            raise ValueError("Please, check the layer number")
    
    
    def prepare_true_data(self, sowing_date, end_date):
        self.ground_truth_yield = self.data_soil[self.data_soil['Horizon_number'] == self.n_layers]['Yield_kg_per_ha']
        self.peak_datetime = pd.to_datetime(end_date)
        
        
        f_name = os.path.join(self.input_path_imag, self.imag_meta_data_f_name)
        
        indexes = indexes_from_coordinates(f_name, self.data_f, self.longit, self.latit, self.profile_id_column_name)
        
        dates, image_ts = pix_timeseries(indexes, initial_date=sowing_date, final_date=end_date, path=self.input_path_imag)
        
        # Filtering the data by the threshold
        self.filtered_ndvi_data, self.filtered_dates = filter_vegetation_ind(self.threshold , image_ts, dates)
    
    
    def run_monica(self, params):
        
        x0, x1, x2, x3, x4, x5, x6 = [round(p, 6) for p in params]
        #                    Base_Temp            Sum_temperature
        params_unpack = [[x0,x0,  x0+2,x0+2   ,x1,x1],[x2,x3,x4,x5,x6, 20.0]]
        
        
        
        crop_dat = self.crop_dat_template.copy()
        cultivar_dat = self.cultivar_dat_template.copy()
        
        crop_dat["BaseTemperature"] = params_unpack[0]
        cultivar_dat["StageTemperatureSum"][0] = params_unpack[1]
        
        
        # Save new parameters
        with open(self.crop_file, 'w', encoding="utf-8") as f:
            json.dump(crop_dat, f, indent=4, ensure_ascii=False)
        with open(self.cultivar_file, 'w', encoding="utf-8") as f:
            json.dump(cultivar_dat, f, indent=4, ensure_ascii=False)
        
        ## MONICA in parallel
        if self.parallel:
            with ThreadPoolExecutor(max_workers=6) as executor:
            #with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(run_monica_for_file, file_id, self.working_directory): file_id
                    for file_id in self.locations_id}
                
                for future in as_completed(futures):
                    file_id = futures[future]
                    try:
                        future.result()
                        #print(f" Done: {file_id}")
                    except Exception as e:
                        print(f" Failed: {file_id} — {e}")
        
        
        else:
            os.chdir(self.working_directory)
            os.environ["MONICA_PARAMETERS"] = "..\\monica-parameters"
            for file_id in self.locations_id:
                result = subprocess.run(["_monica-run.exe", f"sim_{file_id}.json"])
                if result.returncode != 0:
                    raise RuntimeError("MONICA execution failed.")
        
        
        
        r_top = [f"rootDensity_{i}" for i in range(1, 4)]     # 1–3
        r_bot = [f"rootDensity_{i}" for i in range(4, 21)]    # 4–20
        
        
        root1_, root2_, yield_sim, lais, max_lai_dates, sum_tra = [], [], [], [], [], []
        for out in self.locations_id:
            #out_f = f"{self.results_folder}/out_{out}.csv"
            out_f = os.path.join( self.results_folder, f"out_{out}.csv" )
            pd_data = pd.read_csv(out_f, skiprows=1)
            pd_data["Date"] = pd.to_datetime(pd_data["Date"])
            
            yield_sim.append(pd_data["Yield"].iloc[-2])
            lais.append(pd_data[pd_data["Date"].isin(self.filtered_dates)][["LAI"]])
            max_lai_dates.append(pd_data.loc[pd_data["LAI"].idxmax(), "Date"])
            
            # root-density sums #
            root1 = pd_data.loc[pd_data["Date"].isin(self.filtered_dates), r_top].sum(axis=1)
            root2 = pd_data.loc[pd_data["Date"].isin(self.filtered_dates), r_bot].sum(axis=1)
            
            root1_.append(root1)
            root2_.append(root2)
            
            # sum_tra #
            
            
            ### CHANGE HERE TO cum sum in the begginging or not !!!
            pd_data["TraS"] = pd_data["Tra"].cumsum()
            sum_tra_ = pd_data[pd_data["Date"].isin(self.filtered_dates)][["TraS"]]
            sum_tra.append(sum_tra_)
            
        return yield_sim, lais, max_lai_dates, root1_, root2_, sum_tra
    
    
    
    def compute_objective(self, params):
        """
        weights - list containig the weights for each part of the obective function [yield(*), correlation(-1 to 1-), final_date(7 days is a good guess)]
        """
        
        # run MONICA for each location
        yield_sim, lais_sim, max_lai_dates_sim, sum_traa = self.run_monica(params)
        
        ### YIELD
        rmse = np.sqrt(np.mean((np.array(yield_sim) - self.ground_truth_yield.values) ** 2))
        norm_rmse = rmse * self.weights[0]
        
        ### CORRELATIONS - LAI & NDVI
        if self.lai_vi:
            correlations = []
            for i in range(len(lais_sim)):
                lai_vals = lais_sim[i]['LAI'].values
                data_vals = self.filtered_ndvi_data[i]
                corr = np.corrcoef(lai_vals, data_vals)[0, 1]
                #corr = np.corrcoef(np.log1p( lai_vals ), data_vals)[0, 1]   # for log(LAI)
                correlations.append(corr)
        
        else:
            correlations = []
            for i in range(len(sum_traa)):
                lai_vals = sum_traa[i].values
                
                print(lai_vals)
                sys.exit()   # FIX !!
                
                data_vals = self.filtered_ndvi_data[i]
                corr = np.corrcoef(lai_vals, data_vals)[0, 1]
                #corr = np.corrcoef(np.log1p( lai_vals ), data_vals)[0, 1]   # for log(LAI)
                correlations.append(corr)
        
        mean_corr = np.mean(correlations)
        #corr_term = weights[1] * (1 - mean_corr) ** 2
        corr_term = -(self.weights[1] * mean_corr)
        
        if peak:
            ### PEAK
            peak_errors = [abs((self.peak_datetime - sim_peak).days)
                           for sim_peak in max_lai_dates_sim]
            peak_rmse = np.sqrt(np.mean(np.array(peak_errors) ** 2))
            peak_term = peak_rmse * self.weights[2]
        else:
            peak_rmse = 0.
            peak_term = 0.
        
        
        total_cost = norm_rmse + corr_term + peak_term
        
        print("Running objective for params:", [round(p, 4) for p in params] )
        print("yield, corr, peak", rmse , mean_corr , peak_rmse)
        print("Total cost:", total_cost)
        
        #print(norm_rmse , corr_term , peak_term)
        
        #sys.exit()
        return total_cost
        
        
        
        
    def compute_residuals_ML(self, params):
        
        yield_sim, lais_sim, max_lai_dates_sim, root1_, root2_, sum_tra_t = self.run_monica(params)
        
        # Yield residual (normalized)
        yield_resid = (np.array(yield_sim) - self.ground_truth_yield.values) * self.weights[0]
        
        
        
        lai_all = []
        ndwi_all = []
        
        if self.lai_vi:
            for i in range(len(lais_sim)):
                lai_vals = lais_sim[i]['LAI'].values
                data_vals = self.filtered_ndvi_data[i]
                
                #lai_transformed = np.log1p(lai_vals)
                lai_transformed = lai_vals
                
                lai_all.extend(lai_transformed)
                ndwi_all.extend(data_vals)
        
        else:
            for i in range(len(lais_sim)):
                lai_vals = sum_tra_t[i]['TraS'].values
                
                data_vals = self.filtered_ndvi_data[i]
                
                #lai_transformed = np.log1p(lai_vals)
                lai_transformed = lai_vals
                
                lai_all.extend(lai_transformed)
                ndwi_all.extend(data_vals)    
        
        
        corr = np.corrcoef(lai_all, ndwi_all)[0, 1]
        correlation_resid = [  (1 - corr) * self.weights[1]  ]
                
        
        # Peak date residual (days)
        peak_errors = [abs((self.peak_datetime - sim_peak).days)
                       for sim_peak in max_lai_dates_sim]
        peak_rmse = np.sqrt(np.mean(np.array(peak_errors) ** 2))
        peak_resid = peak_rmse * self.weights[2]
        
        
        #mean_corr = np.mean(correlations)
        yield_diff = np.array(yield_sim) - self.ground_truth_yield.values
        rmse_yield = np.sqrt(np.mean(yield_diff ** 2))
        peak_rmse_days = np.sqrt(np.mean(np.array(peak_errors) ** 2))
        print()
        print("Yield RMSE (kg/ha):", round(rmse_yield, 2), round( (rmse_yield * self.weights[0]) , 2)  )
        #print("Mean correlation (LAI vs NDVI):", round(mean_corr, 3),  round(mean_corr * self.weights[1] , 3)  )
        print("Correlation weight (LAI vs NDVI):", round(correlation_resid[0], 3)  )
        print("Correlation:", round( corr ,4) )
        print("Peak LAI RMSE (days):", round(peak_rmse_days, 2) )
        
        if peak:
            return np.concatenate([
                yield_resid,            # list of yield residuals
                correlation_resid,           # single corr residual
                [peak_resid]])          # single peak residual
        else:
            return np.concatenate([
                yield_resid,            # list of yield residuals
                correlation_resid])          # single corr residual



if __name__ == '__main__':
    #print("Starting class")
    
    fitter = monica_fitting(data_f, input_path_imag_1, working_directory, crop_file, cultivar_file, results_folder, threshold, weights, file_name_meta, profile_id_column_name)
    fitter.prepare_true_data(sowing_date, end_date)
    
    if save_output_stop:
        
        fitter.filtered_ndvi_dataT = fitter.filtered_ndvi_data.T
        df = pd.DataFrame(fitter.filtered_ndvi_dataT)
        
        df.insert(0, "Date", fitter.filtered_dates)
        df.to_csv( os.path.join(save_output_path , rf"_veg_ind_{ind_tag}_output.csv"), index=False)
        
        
        
        with open(path_param, 'r') as f:
            params = json.load(f)
        
        yield_sim_, lais_, max_lai_dates_, root1, root2, sum_tra = fitter.run_monica(params)
        
        yield_sim_ = [float(x) for x in yield_sim_]
        
        print("locations id:", fitter.locations_id)
        print("Yields_sim:  ", yield_sim_)
        print("Yields_true: ", fitter.ground_truth_yield.values)
        print("RMSE:", rmse_(yield_sim_ , fitter.ground_truth_yield.values) )
        
        series_list_vi = [df["LAI"] if isinstance(df, pd.DataFrame) else df for df in lais_]
        combined_df_vi = pd.concat( series_list_vi , axis=1)
        combined_df_vi.to_csv(  os.path.join(save_output_path , r"_combined_lai.csv"), index=False)
        
        # Combine the top‐layer root sums
        df_root1 = pd.concat(root1, axis=1)
        df_root1.columns = fitter.locations_id
        df_root1.insert(0, "Date", fitter.filtered_dates)
        df_root1.to_csv(  os.path.join(save_output_path , "_root1_timeseries.csv"), index=False)
        
        # Combine the deep‐layer root sums
        df_root2 = pd.concat(root2, axis=1)
        df_root2.columns = fitter.locations_id
        df_root2.insert(0, "Date", fitter.filtered_dates)
        df_root2.to_csv(  os.path.join( save_output_path , "_root2_timeseries.csv"), index=False)
        
        # Sum Tra
        df_sum_tra = pd.concat(sum_tra, axis=1)
        df_sum_tra.columns = fitter.locations_id
        df_sum_tra.insert(0, "Date", fitter.filtered_dates)
        df_sum_tra.to_csv(  os.path.join(save_output_path , "_sum_tra_timeseries.csv"), index=False)
        
        sys.exit()
    
    
    else:
        print("Starting minim..")
        
        if ML:
            
            diff_ = 0.1
            
            
            result = least_squares(fitter.compute_residuals_ML,
                                   x0=params_ini,
                                   bounds=([b[0] for b in bounds], [b[1] for b in bounds]),
                                   verbose=2,
                                   xtol=1e-3, ftol=1e-3, max_nfev=70, diff_step=diff_)
            print("\nOptimization done")
            print("Best parameters found:", "[" + ", ".join(f"{x:.4f}" for x in result.x) + "]"  )
            print("Final cost (sum of squared residuals):", np.sum(result.fun ** 2))
            
            
        else:
            result = minimize(fun=fitter.compute_objective,
                      x0=params_ini,
                      #args=(),  # or args=(your_weights,) if you make it configurable
                      method='L-BFGS-B',
                      bounds=bounds,
                      options={
                          'disp': True,
                          'eps': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],        # You can fine-tune this
                          'maxiter': 70})
        
            print("\n Optimization done")
            print("Best parameters found:", result.x)
            print("Final objective value:", result.fun/len(fitter.locations_id))








