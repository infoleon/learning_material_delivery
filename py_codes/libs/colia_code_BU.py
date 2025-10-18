from . import models as m_models
from swn import models as swn_models
from django.forms import modelformset_factory
from .forms import *
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse, reverse_lazy
from django.http import HttpResponse, HttpResponseRedirect, HttpRequest, HttpResponseBadRequest, JsonResponse
from django.contrib.gis.geos import Polygon
from django.contrib.gis.db.models.functions import Transform
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.template.loader import render_to_string
from django.apps import apps

from .run_consumer_swn import run_consumer
from .run_producer import run_producer
from pathlib import Path
from . import monica_io3_swn
from .climate_data.lat_lon_mask import lat_lon_mask

from netCDF4 import Dataset, MFDataset
import numpy as np
from datetime import datetime, timedelta
from cftime import num2date, date2num, date2index

import pytz
import json
import zmq
import csv
import bisect
# create a new monica env
from . import climate
from .process_weather_data import *
from datetime import datetime, timedelta


#-------------------------- Monica Interface --------------------------#
def add_workstep(request):
    workstep_type = request.GET.get('workstep_type')
    if workstep_type == 'sowing':
        return WorkstepSowingForm()
    elif workstep_type == 'harvest':
        return WorkstepHarvestForm()
    elif workstep_type == 'tillage':
        return WorkstepTillageForm()
    elif workstep_type == 'mineral_fertilization':
        return WorkstepMineralFertilizationForm()
    elif workstep_type == 'organic_fertilization':
        return WorkstepOrganicFertilizationForm()
    else:
        return None



# all relevant climate variables. The key is already the the correct key for  MONICA climate jsons
CLIMATE_VARIABLES = { 
    '3': 'tasmin',
    '4': 'tas',
    '5': 'tasmax',
    '6': 'pr',
    '8': 'rsds',
    '9': 'sfcWind',
    '12': 'hurs'
    }

def get_lat_lon_as_index(lat, lon):
    """
    Returns the index of the closest lat, lon to the given lat, lon in the netCDF grid. Used for matching differntly scaled grids.
    """
    lats = lat_lon_mask['lat']
    
    lons = lat_lon_mask['lon']
    lat_idx = 0
    lon_idx = 0
    for i, lat_ in enumerate(lats):
        if lat_ < lat:
            lat_idx = i
            break
    
    for j, lon_ in enumerate(lons):
        if lon_ > lon:
            lon_idx = j
            break
   
    if (lats[lat_idx] + lats[lat_idx-1]) / 2 < lat:
        print("lat is in if")
        lat_idx = lat_idx - 1

    if (lons[lon_idx] + lons[lon_idx-1]) / 2 > lon:
        print("lon is in if")
        lon_idx = lon_idx - 1

    print('lat', lats[lat_idx], 'lon_idx', lons[lon_idx])
    return (lat_idx, lon_idx)


def get_climate_data_as_json(start_date, end_date, lat_idx, lon_idx):
    """Returns the climate data as json using monica's keys for the given start and end date and the given lat and lon index"""
    print("get_climate_data_as_json", start_date, end_date, lat_idx, lon_idx)
    # opening with MFDataset does not work, because time is not an unlimited dimension in the NetCDF files
    start = datetime.now()
    climate_json = { 
        '3': [],
        '4': [],
        '5': [],
        '6': [],
        '8': [],
        '9': [],
        '12': [],
        }
    climate_data_path = Path(__file__).resolve().parent.joinpath('climate_netcdf')
    for year in range(start_date.year, end_date.year + 1):
        for key, value in CLIMATE_VARIABLES.items():
            
            file_path = f"{climate_data_path}/zalf_{value.lower()}_amber_{year}_v1-0.nc"
            print("filepath: ", file_path,  "getting key:", key)
            nc = Dataset(file_path, 'r')
 
            start_idx = 0
            end_idx = len(nc['time']) + 1
            if year == start_date.year:
                start_idx = date2index(start_date, nc['time'])
            if year == end_date.year:
                end_idx = date2index(end_date, nc['time']) +1

            values = nc.variables[value][start_idx:end_idx, lat_idx, lon_idx]
            values = values.tolist()

            climate_json[key].extend(values)
            nc.close()

            print(year, value, key)
    print('Time elapsed in get_climate_data_as_json: ', datetime.now() - start)
    return climate_json

### MONICA VIEWS ###
def create_monica_env(
    species_id, 
    cultivar_id, 
    soil_profile_id,
    start_date,
    end_date,
    lat_idx=None,
    lon_idx=None,
    lat = 52.8,
    lon = 13.8,
    slope = 0,
    height_nn = 0,
    n_deposition = 30,
    user_crop_parameters_id = 1,
    user_environment_parameters_id = 1,
    user_soil_moisture_parameters_id = 1,
    user_soil_tempereature_parametes_id = 1,
    user_soil_transport_parameters_id = 1,
    user_soil_organic_parameters_id = 1
    ):
    # TODO: pull this out of here and catch errors
    if lat_idx is None or lon_idx is None:
        lat_idx, lon_idx = m_models.DWDGridAsPolygon.get_idx(lat, lon)
    climate_data = get_climate_data_as_json(start_date, end_date, lat_idx, lon_idx)
    # temporary replacements from file crop_site_sim2.json until available from database 
    simj = {
        "debug?": True,
        "UseSecondaryYields": True,
        "NitrogenResponseOn": True,
        "WaterDeficitResponseOn": True,
        "EmergenceMoistureControlOn": True,
        "EmergenceFloodingControlOn": True,
        "UseNMinMineralFertilisingMethod": True,
        "NMinUserParams": {
            "min": 40,
            "max": 120,
            "delayInDays": 10
            },
        "NMinFertiliserPartition": m_models.MineralFertiliser.objects.get(id=3).to_json(),
        "JulianDayAutomaticFertilising": 89,
        "UseAutomaticIrrigation": False,
        "AutoIrrigationParams": {
            "irrigationParameters": {
                "nitrateConcentration": [0,"mg dm-3"],
                "sulfateConcentration": [0,"mg dm-3"]
            },
            "amount": [17,"mm"],
            "threshold": 0.35
        }
    }
    
    cropRotation = [{
        "worksteps": [{
            "date": "0000-10-13",
            "type": "Sowing",
            "crop": {
                # "is-winter-crop": True, # TODO is winter-crop is probably not required!!!
                "cropParams": {
                    "species": {
                    "=": m_models.SpeciesParameters.objects.get(id=species_id).to_json()
                    },
                    "cultivar": {
                    "=": m_models.CultivarParameters.objects.get(id=cultivar_id).to_json()
                    }
                },
                "residueParams": m_models.CropResidueParameters.objects.get(species_parameters=species_id).to_json()
            }
        }, {
            "type": "Harvest",
            "date": "0001-05-21"
        }]
    }]

    cropRotations = None

    # events define the output.
    events = [
        "daily",
        [
            "Date",
            "Crop",
            "Stage",
            "ETa/ETc",
            "AbBiom",
            [
            "OrgBiom",
            "Leaf"
            ],
            [
            "OrgBiom",
            "Fruit"
            ],
            "Yield",
            "LAI",
            "Precip",
            [
            "Mois",
            [
                1,
                20
            ]
            ],
            [
            "Mois",
            [
                1,
                10,
                "AVG"
            ]
            ],
            [
            "SOC",
            [
                1,
                3
            ]
            ],
            "Tavg",
            "Globrad"
        ],
        "crop",
        [
            "CM-count",
            "Crop",
            [
            "Yield",
            "LAST"
            ],
            [
            "Date|sowing",
            "FIRST"
            ],
            [
            "Date|harvest",
            "LAST"
            ]
        ],
        "yearly",
        [
            "Year",
            [
            "N",
            [
                1,
                3,
                "AVG"
            ],
            "SUM"
            ],
            [
            "RunOff",
            "SUM"
            ],
            [
            "NLeach",
            "SUM"
            ],
            [
            "Recharge",
            "SUM"
            ]
        ],
        "run",
        [
            [
            "Precip",
            "SUM"
            ]
        ]
    ]
    # end of replacement -------------------------------------------
    
    debugMode = True
    
    soil_horizons = swn_models.BuekSoilProfileHorizon.objects.filter(bueksoilprofile=soil_profile_id, obergrenze_m__gte=0).order_by('horizont_nr')
    soil_parameters = [horizon.to_json() for horizon in soil_horizons]
    
    siteParameters = {
        "Latitude": lat,
        "Slope": slope,
        "HeightNN": [height_nn, "m"],
        "NDeposition": [n_deposition,"kg N ha-1 y-1"],
        "SoilProfileParameters": soil_parameters
    }
    cpp = {
    "type": "CentralParameterProvider",
    "userCropParameters": m_models.UserCropParameters.objects.get(id=user_crop_parameters_id).to_json(),
    "userEnvironmentParameters": m_models.UserEnvironmentParameters.objects.get(id=user_environment_parameters_id).to_json(),
    "userSoilMoistureParameters": m_models.UserSoilMoistureParameters.objects.get(id=user_soil_moisture_parameters_id).to_json(),
    "userSoilTemperatureParameters": m_models.SoilTemperatureModuleParameters.objects.get(id=user_soil_tempereature_parametes_id).to_json(),
    "userSoilTransportParameters": m_models.UserSoilTransportParameters.objects.get(id=user_soil_transport_parameters_id).to_json(),
    "userSoilOrganicParameters": m_models.UserSoilOrganicParameters.objects.get(id=user_soil_organic_parameters_id).to_json(),
    "simulationParameters": simj,
    "siteParameters": siteParameters
    }


    # print('available_climate_data', available_climate_data)
    print("check 1")
    
    print("CLIMATE DATA START DATE ", start_date.date().isoformat())
    print("check 2")
    climate_json = {
        "type": "DataAccessor",
        "data": climate_data,
        "startDate": start_date.date().isoformat(),
        "endDate": end_date.date().isoformat(),
      }
    
    print("check 3")
    env = {
        "type": "Env",
        "debugMode": debugMode,
        "params": cpp,
        "cropRotation": cropRotation,
        "cropRotations": cropRotations,
        "events": events,
        # "climateData": json.dumps(climate_json)
        "climateData": climate_json
    }
    print("check 4")

    print("check 4b: \n", env)

    return env

# get options for cultivar parameters selectbox in monica_hohenfinow_db.html
# todo: DELETE THIS FUNCTION used in monica_crop.js!!!
def get_cultivar_parameters(request, id):       
    cultivars = m_models.CultivarParameters.objects.filter(species_parameters=id).order_by('name')
    cultivar_list = []
    for cultivar in cultivars:
        if cultivar.name != '':
            cultivar_list.append({
                'id': cultivar.id,
                'name': cultivar.name,
            })
        else:
            cultivar_list.append({
                'id': cultivar.id,
                'name': 'default',
            })

    # selected species plant density
    species = m_models.SpeciesParameters.objects.get(id=id)
    
    return JsonResponse({'cultivars': cultivar_list, 'plant_density': species.plant_density})

def monica_calc_w_params_from_db(request):
    start_time = datetime.now()
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_field_id = data.get('userFieldId')
            species_id = data.get('speciesId')
            cultivar_id = data.get('cultivarId')
            soil_profile_id = data.get('soilProfileId')
            start_date_str = data.get('startDate')
            end_date_str = data.get('endDate')

            start_date = datetime.strptime(start_date_str, "%m/%d/%Y")
            end_date = datetime.strptime(end_date_str, "%m/%d/%Y")   

            print("TRY TO GET WEATHER DATA: ", user_field_id)

            weather_coords = []
            print("Cherckpoint 0")
            if not swn_models.UserField.objects.get(id=user_field_id).weather_grid_points:
                print("Cherckpoint 1")
                weather_coords = swn_models.UserField.objects.get(id=user_field_id).get_weather_grid_points()
            else:
                weather_coords = json.loads(swn_models.UserField.objects.get(id=user_field_id).weather_grid_points)
            

            envs = []
            for coord in weather_coords["weather_indices"]:
                lat = coord['lat']
                lon = coord['lon']
                lat_idx = coord['lat_idx']
                lon_idx = coord['lon_idx']
                if coord['is_valid']:
                    # climate_json = get_climate_data_as_json(start_date, end_date, lat_idx, lon_idx)   
                    envs.append(create_monica_env(species_id=species_id, cultivar_id=cultivar_id, soil_profile_id=soil_profile_id, start_date=start_date, end_date=end_date, lat_idx=lat_idx, lon_idx=lon_idx, lat=lat, lon=lon))                 
                else:
                    continue
                
            print("check 5")
            msgs = []
            for env in envs:
                context = zmq.Context()
                socket = context.socket(zmq.PUSH)
                socket.connect("tcp://swn_monica:6666")
                print("check 6")
                # print(env)
                socket.send_json(env)
                print("check 7")


                file_path = Path(__file__).resolve().parent
                with open(f'{file_path}/env_from_db.json', 'w') as _: 
                    json.dump(env, _)
                    print("check 8")
                msg = run_consumer()
                msg = msg_to_json(msg)
                print("check 9")

                msgs.append(msg)

            end_time = datetime.now()
            print("Time elapsed in monica_calc: ",  end_time - start_time)
            # TODO oly the first result is sent to front end
            return JsonResponse({'result': 'success', 'msg': msgs[0]})
        
        except Exception as e:
            print('Error', e)
            return JsonResponse({'result': 'error', 'msg': 'Invalid request'})
    
# layerAggOp
OP_AVG = 0
OP_MEDIAN = 1
OP_SUM = 2
OP_MIN = 3
OP_MAX = 4
OP_FIRST = 5
OP_LAST = 6
OP_NONE = 7
OP_UNDEFINED_OP_ = 8


ORGAN_ROOT = 0
ORGAN_LEAF = 1
ORGAN_SHOOT = 2
ORGAN_FRUIT = 3
ORGAN_STRUCT = 4
ORGAN_SUGAR = 5
ORGAN_UNDEFINED_ORGAN_ = 6


def msg_to_json(msg):
    """
    the json output of Monica is processed in this function so that every output 
    is a flat array with the length of the base array e.g. of dates for each output. 
    This is applied to all outputs such as 'daily', 'monthly, 'crop' etc."""
    # aggregation constants as dictionary from monica_io3.py
    aggregation_constants = {
        0: "AVG",
        1: "MEDIAN",
        2: "SUM",
        3: "MIN",
        4: "MAX",
        5: "FIRST",
        6: "LAST",
        7: "NONE",
        8: "UNDEFINED_OP"
    }
    # organ constants as dictionary from monica_io3.py
    organ_constants = {
        0: "ROOT",
        1: "LEAF",
        2: "SHOOT",
        3: "FRUIT",
        4: "STRUCT",
        5: "SUGAR",
        6: "UNDEFINED_ORGAN"
    }


    print("msg_to_json")
    processed_msg = {}
    for_chart = {}
    for data_ in msg.get("data", []):
        results = data_.get("results", [])
        orig_spec = data_.get("origSpec", "")
        output_ids = data_.get("outputIds", [])

        orig_spec = orig_spec.replace("\"", "")
        
        for_chart[orig_spec] = {}
        for output_id, result_list in zip(output_ids, results):
            output_id["result"] = result_list
            try:
                output_id["jsonInput"] = json.loads(output_id["jsonInput"])
            except:
                pass
            
            if output_id["fromLayer"] == output_id["toLayer"] == -1:
                output_id["result_dict"] = {}
                name = output_id["name"]
                if output_id["organ"] != 6:
                    name = name + "_" + organ_constants[output_id["organ"]]
                    
                output_id["result_dict"][name] = result_list

                for_chart[orig_spec][name] = result_list
            elif output_id["layerAggOp"] != 7:
                output_id["result_dict"] = {}
                output_id["result_dict"][f"{output_id['name']}_{aggregation_constants[output_id['layerAggOp']]}"] = result_list

                for_chart[orig_spec][f"{output_id['name']}_{aggregation_constants[output_id['layerAggOp']]}"] = result_list
            elif (output_id["fromLayer"] != output_id["toLayer"]) and (output_id["layerAggOp"] == 7):
                # no aggregation of layers, but calculations for several layers
                output_id["result_dict"] = {}
                try:
                    for i in range(output_id["fromLayer"], output_id["toLayer"]+1):
                        # print("I: ", i)
                        output_id["result_dict"][f"{output_id['name']}_{i+1}"] = []
                        for j in range(len(result_list)): 
                            output_id["result_dict"][f"{output_id['name']}_{i+1}"].append(result_list[j][i])
                        for_chart[orig_spec][f"{output_id['name']}_{i+1}"] = output_id["result_dict"][f"{output_id['name']}_{i+1}"]
                except:
                    output_id["result_dict"]["error"] = "Error in processing results"
                    
        processed_msg[orig_spec] = {
            "output_ids": output_ids,     
        }
           
    return for_chart

def export_monica_result_to_csv(msg):

    file_path = Path(__file__).resolve().parent
    
    file_name = 'monica_result_.csv'
    file_path = Path.joinpath(file_path, 'monica_csv_exports/', file_name)
    with open(file_path, 'w', newline='') as _:
        writer = csv.writer(_, delimiter=",")

        for data_ in msg.get("data", []):
            results = data_.get("results", [])
            orig_spec = data_.get("origSpec", "")
            output_ids = data_.get("outputIds", [])

            if len(results) > 0:
                writer.writerow([orig_spec.replace("\"", "")])
                for row in monica_io3_swn.write_output_header_rows(output_ids,
                                                                include_header_row=True,
                                                                include_units_row=True,
                                                                include_time_agg=False):
                    writer.writerow(row)

                for row in monica_io3_swn.write_output(output_ids, results):
                    writer.writerow(row)

            writer.writerow([])

def get_site_params_height(polygon):
    polygon_25832 = Transform(polygon, srid=25832)

    dem_data_within_polygon = m_models.DigitalElevationModel.objects.filter(
        grid_file__intersects=polygon_25832
    )

    if dem_data_within_polygon.count() == 0:
        return None
    else:
        for dem in dem_data_within_polygon:
            print("DEM: ", dem)
            # dem_data = dem.grid_file.read()
            # dem_data = np.flipud(dem_data)
            # dem_data = np.ma.masked_where(dem_data == dem.nodata_value, dem_data)
            # dem_data = np.ma.masked_where(dem_data < 0, dem_data)
            # dem_data = np.ma.masked_where(dem_data > 1000, dem_data)
            # height = np.mean(dem_data)
            # return height

def monica_form(request):
    cp_form = CultivarAndSpeciesSelectionForm()

    # dates
    # start and enddate according to the available weather data
    start_date = "01.01.2007"
    end_date = "31.12.2023"
    now = datetime.now()
    set_start_date = f"{str(now.day)}.{str(now.month)}.{str(now.year - 1)}"
    print("SetStartDate", set_start_date, type(set_start_date))
    set_end_date = now.strftime("%d.%m.%Y")
    
    form = {
        'date_picker':{
            'start_date': start_date,
        'end_date': end_date,
        'set_start_date': set_start_date,
        'set_end_date': set_end_date,
        },
        'cultivar_form': cp_form
    }
    return render(request, 'monica/monica_form.html', form)
    
# def crop_residue_parameters(request, id):
#     residue = get_object_or_404(CropResidueParameters, pk=id)

#     if request.method == 'POST':
#         form = CropResidueParametersForm(request.POST, instance=residue)
#         if form.is_valid():
#             instance = form.save(commit=False)
#             if 'save_as_new' in request.POST:
#                 instance.pk = None 
#             elif 'save' in request.POST and instance.default:
#                 return JsonResponse({'success': False, 'errors': 'Cannot modify the default species parameters. Please use save as new.'})
#             instance.save()
#             return JsonResponse({'success': True})
            
#         else:
#             return JsonResponse({'success': False, 'errors': form.errors})
#     else:
#         form = CropResidueParametersForm(instance=residue)
#         modal_title = 'Modify Crop Residue Parameters'
#         modal_save_button = 'Save Crop Residue Parameters'
#         modal_save_as_button = 'Save as New Crop Residue Parameters'
#         data_action_url = 'crop_residue_parameters/' + str(id) + '/'
#         context = {
#             'form': form,
#             'modal_title': modal_title,
#             'modal_save_button': modal_save_button,
#             'modal_save_as_button': modal_save_as_button,
#             'data_action_url': data_action_url,
#         }
#         return render(request, 'monica/modify_parameters_modal.html', context)

def get_parameter_options(request, parameter_type, id=None):
    if parameter_type == 'soil-moisture-parameters':
        options = UserSoilMoistureParameters.objects.values('id', 'name')
    elif parameter_type == 'soil-organic-parameters':
        options = UserSoilOrganicParameters.objects.values('id', 'name')
    elif parameter_type == 'soil-temperature-parameters':
        options = SoilTemperatureModuleParameters.objects.values('id', 'name')
    elif parameter_type == 'soil-transport-parameters':
        options = UserSoilTransportParameters.objects.values('id', 'name')
    elif parameter_type == 'species-parameters':
        options = SpeciesParameters.objects.values('id', 'name')
    elif parameter_type == 'cultivar-parameters':
        print("GETTING CULTIVAR PARAMETERS", id)
        if id is not None:
            options = CultivarParameters.objects.filter(species_parameters_id=id).values('id', 'name')
            if options.count() == 0:
                options = CultivarParameters.objects.values('id', 'name')
        else:
            options = CultivarParameters.objects.values('id', 'name')
    elif parameter_type == 'crop-residue-parameters':
        if id is not None:
            options = CropResidueParameters.objects.filter(species_parameters_id=id).values('id', 'name')
            if options.count() == 0:
                options = CropResidueParameters.objects.values('id', 'name')
        else:
            options = CropResidueParameters.objects.values('id', 'name')
    else:
        options = []

    return JsonResponse({'options': list(options)})
    

def modify_model_parameters(request, model_name, id):
    MODEL_FORM_MAPPING = {
        'species-parameters': {
            'model': SpeciesParameters,
            'form': SpeciesParametersForm,
            'modal_title': 'Modify Species Parameters',
            'modal_save_button': 'Save Species Parameters',
            'modal_save_as_button': 'Save as New Species Parameters',
        },
        'cultivar-parameters': {
            'model': CultivarParameters,
            'form': CultivarParametersForm,
            'modal_title': 'Modify Cultivar Parameters',
            'modal_save_button': 'Save Cultivar Parameters',
            'modal_save_as_button': 'Save as New Cultivar Parameters',
        },
        'crop-residue-parameters': {
            'model': CropResidueParameters,
            'form': CropResidueParametersForm,
            'modal_title': 'Modify Crop Residue Parameters',
            'modal_save_button': 'Save Crop Residue Parameters',
            'modal_save_as_button': 'Save as New Crop Residue Parameters',
        },
        'organic_fertiliser': {
            'model': OrganicFertiliser,
            'form': OrganicFertiliserForm,
            'modal_title': 'Modify Organic Fertiliser',
            'modal_save_button': 'Save Organic Fertiliser',
            'modal_save_as_button': 'Save as New Organic Fertiliser',
        },
        'mineral_fertiliser': {
            'model': MineralFertiliser,
            'form': MineralFertiliserForm,
            'modal_title': 'Modify Mineral Fertiliser',
            'modal_save_button': 'Save Mineral Fertiliser',
            'modal_save_as_button': 'Save as New Mineral Fertiliser',
        },
        'user_crop_parameters': {
            'model': UserCropParameters,
            'form': UserCropParametersForm,
            'modal_title': 'Modify Crop Parameters',
            'modal_save_button': 'Save Crop Parameters',
            'modal_save_as_button': 'Save as New Crop Parameters',
        },
        'user_environment_parameters': {
            'model': UserEnvironmentParameters,
            'form': UserEnvironmentParametersForm,
            'modal_title': 'Modify Environment Parameters',
            'modal_save_button': 'Save Environment Parameters',
            'modal_save_as_button': 'Save as New Environment Parameters',
        },
        'soil-moisture-parameters': {
            'model': UserSoilMoistureParameters,
            'form': UserSoilMoistureParametersForm,
            'modal_title': 'Modify Soil Moisture Parameters',
            'modal_save_button': 'Save Soil Moisture Parameters',
            'modal_save_as_button': 'Save as New Soil Moisture Parameters',
        },
        'soil-organic-parameters': {
            'model': UserSoilOrganicParameters,
            'form': UserSoilOrganicParametersForm,
            'modal_title': 'Modify Soil Organic Parameters',
            'modal_save_button': 'Save Soil Organic Parameters',
            'modal_save_as_button': 'Save as New Soil Organic Parameters',
        },
        'soil-temperature-parameters': {
            'model': SoilTemperatureModuleParameters,
            'form': SoilTemperatureModuleParametersForm,
            'modal_title': 'Modify Soil Temperature Parameters',
            'modal_save_button': 'Save Soil Temperature Parameters',
            'modal_save_as_button': 'Save as New Soil Temperature Parameters',
        },
        'soil-transport-parameters': {
            'model': UserSoilTransportParameters,
            'form': UserSoilTransportParametersForm,
            'modal_title': 'Modify Soil Transport Parameters',
            'modal_save_button': 'Save Soil Transport Parameters',
            'modal_save_as_button': 'Save as New Soil Transport Parameters',
        }
    }

    if model_name not in MODEL_FORM_MAPPING:
        return JsonResponse({'success': False, 'errors': 'Invalid model name'})
    
    model_info = MODEL_FORM_MAPPING[model_name]
    model_class = model_info['model']
    form_class = model_info['form']

    if model_name == 'crop_residue_parameters':
        obj = get_object_or_404(model_class, species_parameters=id)
    else:
        obj = get_object_or_404(model_class, pk=id)

    if request.method == 'POST':
        form = form_class(request.POST, instance=obj)
        if form.is_valid():
            instance = form.save(commit=False)
            if 'save_as_new' in request.POST:
                instance.pk = None 
            elif instance.is_default and 'save_as_new' not in request.POST:
                return JsonResponse({'success': False, 'errors': 'Cannot modify the default species parameters. Please use save as new.'})
            instance.save()
            print("New primary key? :", instance.pk)
            return JsonResponse({'success': True, 'new_id': instance.pk})
            
        else:
            return JsonResponse({'success': False, 'errors': form.errors})
    else:
        form = form_class(instance=obj)
        context = {
            'form': form,
            'modal_title': model_info['modal_title'],
            'modal_save_button': model_info['modal_save_button'],
            'modal_save_as_button': model_info['modal_save_as_button'],
            'data_action_url': f'{model_name}/{id}/',
        }
        return render(request, 'monica/modify_parameters_modal.html', context)

# @login_required
def monica_model(request):

    coordinate_form = CoordinateForm()
   
    workstep_selector_form =  WorkstepSelectorForm()
    workstep_sowing_form = WorkstepSowingForm()
    workstep_harvest_form = WorkstepHarvestForm()
    workstep_tillage_form = WorkstepTillageForm()
    workstep_mineral_fertilization_form = WorkstepMineralFertilizationForm()
    workstep_organic_fertilization_form = WorkstepOrganicFertilizationForm()

    # species_form = SpeciesParametersForm()  
    # cultivar_form = CultivarParametersForm()
    # residue_form = CropResidueParametersForm()
    
    sim_settings_instance = m_models.UserSimulationSettings.objects.get(name='default')
    simulation_settings_form = UserSimulationSettingsForm(instance=sim_settings_instance)

    user_soil_moisture_select_form = UserSoilMoistureInstanceSelectionForm()
    user_soil_organic_select_form = UserSoilOrganicInstanceSelectionForm()
    soil_temperature_module_selection_form = SoilTemperatureModuleInstanceSelectionForm()
    user_soil_transport_parameters_selection_form = UserSoilTransportParametersInstanceSelectionForm()




    ## POST LOGIC
    if request.method == 'POST':
        print("Request POST: ", request.POST)
        if 'save_simulation_settings' in request.POST or 'save_as_simulation_settings' in request.POST:
            simulation_settings_form = UserSimulationSettingsForm(request.POST)
            if simulation_settings_form.is_valid():
                
                if 'save_simulation_settings' in request.POST:
                    if sim_settings_instance.name == 'default' and sim_settings_instance.default and sim_settings_instance.user is None:
                        messages.error(request, "Cannot modify the default settings.")
                    else:
                        # Update existing settings
                        simulation_settings_instance = UserSimulationSettings.objects.get(id=request.POST.get('id'))
                        for field in simulation_settings_form.cleaned_data:
                            setattr(sim_settings_instance, field, simulation_settings_form.cleaned_data[field])
                        simulation_settings_instance.save()
                        messages.success(request, "Settings updated successfully.")
                elif 'save_as_simulation_settings' in request.POST:
                    # Save as new settings
                    new_name = request.POST.get('new_name')
                    if new_name:
                        new_settings = simulation_settings_form.save(commit=False)
                        new_settings.name = new_name
                        new_settings.user = request.user
                        new_settings.save()
                        messages.success(request, "Settings saved as new successfully.")
                    else:
                        messages.error(request, "Please provide a new name for the settings.")
            else:
                messages.error(request, "There was an error with the form.")

    
    context = {
        'coordinate_form': coordinate_form,
        #'cultivar_parameters_form': cultivar_parameters_form,
        'simulation_settings_form': simulation_settings_form,
        'workstep_selector_form': workstep_selector_form,
        'workstep_sowing_form': workstep_sowing_form,
        'workstep_harvest_form': workstep_harvest_form,
        'workstep_tillage_form': workstep_tillage_form,
        'workstep_mineral_fertilization_form': workstep_mineral_fertilization_form,
        'workstep_organic_fertilization_form': workstep_organic_fertilization_form,
        # 'species_form': species_form,
        # 'cultivar_form': cultivar_form,
        # 'residue_form': residue_form,

        'user_soil_moisture_select_form': user_soil_moisture_select_form,
        'user_soil_organic_select_form': user_soil_organic_select_form,
        'soil_temperature_module_selection_form': soil_temperature_module_selection_form, 
        'user_soil_transport_parameters_selection_form': user_soil_transport_parameters_selection_form,


    }
    return render(request, 'monica/monica_model.html', context)


