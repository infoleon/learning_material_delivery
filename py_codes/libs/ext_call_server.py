


# -*- coding: utf-8 -*-


import os
import subprocess as sub
import concurrent.futures
import csv
import threading

import sys






class call_server_scripts:
    def __init__(self):
        





if __name__ == "__main__":
    
    path      =  r"D:\_Trabalho\_Publicacoes_nossas\Inv_Mod_SHP\__MONICA_New\projects\Work_single_run" + os.sep
    path      =  r"D:\_Trabalho\_Publicacoes_nossas\Inv_Mod_SHP\__MONICA_New\projects\Work_single_run\2nd_soils_sim" + os.sep
    exe       =  r"monica-run.exe"
    meta_file =  r"_meta_organizer.csv"
    env_path  =  r".\monica-parameters"
    
    run_monica_batch(path, exe, meta_file, env_path)













