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



import sys
import csv
import os
import json
import time

import zmq


#print("pyzmq version: ", zmq.pyzmq_version(), " zmq version: ", zmq.zmq_version())

from . import monica_io3
#print("path to monica_io: ", monica_io.__file__)

def run_consumer(result_thread):
    "collect data from workers"
    
    
    #path_to_output_dir = None
    leave_after_finished_run = False
    server = {"server": None, "port": None}
    shared_id = None
    
    
    #leave_after_finished_run = False
    
    config = {
        "port": server["port"] if server["port"] else "7777",
        #"server": server["server"] if server["server"] else "login01.cluster.zalf.de",#"localhost", 
        "server": server["server"] if server["server"] else "localhost",
        "shared_id": shared_id,
        #"out": path_to_output_dir if path_to_output_dir else os.path.join(os.path.dirname(__file__), './'),
        "leave_after_finished_run": leave_after_finished_run}
    
    if len(sys.argv) > 1 and __name__ == "__main__":
        for arg in sys.argv[1:]:
            k,v = arg.split("=")
            if k in config:
                if k == "leave_after_finished_run":
                    if (v == 'True' or v == 'true'): 
                        config[k] = True
                    else: 
                        config[k] = False
                else:
                    config[k] = v
                    
    #print("consumer config:", config)
    
    context = zmq.Context()
    if config["shared_id"]:
        socket = context.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, config["shared_id"])
    else:
        socket = context.socket(zmq.PULL)
    socket.connect("tcp://" + config["server"] + ":" + config["port"])
    
    #socket.RCVTIMEO = 1000
    def process_message(msg):
        
        if not hasattr(process_message, "wnof_count"):
            process_message.received_env_count = 0
            
        #print("c: received work result ", process_message.received_env_count, " customId: ", str(msg.get("customId", "")))
        
        process_message.received_env_count += 1
        
        # check if message has errors
        if msg.get("errors", []):
            print("c: received errors: ", msg["errors"])
        
        id_number = msg.get("customId", "")
        
        output_data = []
        for data_ in msg.get("data", []):
            results = data_.get("results", [])
            #orig_spec = data_.get("origSpec", "")
            output_ids = data_.get("outputIds", [])
            output_data += results[0]
        
        # cleanning the output
        return output_data, id_number
            
        #     if len(results) > 0:
        #         for row in monica_io3.write_output_header_rows(output_ids,
        #                                                       include_header_row=True,
        #                                                       include_units_row=True,
        #                                                       include_time_agg=False):
        #             output_data.append(row)
        #         for row in monica_io3.write_output(output_ids, results):
        #             output_data.append(row)
        #     #if config["leave_after_finished_run"] == True :
        #     #    leave = True
        # return output_data, id_number
    
    out_res = []
    results_num = 0
    leave = False
    while not leave:
        try:
            msg = socket.recv_json()
            #monica_time = time.perf_counter()
            #start_time = time.perf_counter()
            result , id_number = process_message(msg)
            out_res.append([result , id_number[:2]])
            results_num += 1
            
            # Leave?
            if results_num >= id_number[1]:
                #print(results_num)
                leave = True
            
            #end_time = time.perf_counter()
            #print("cons: i:", id_number[0], "e-s:", end_time - start_time, "seconds", "send+monica+receive:", monica_time - id_number[2])
            
        except:
            print(sys.exc_info())
            continue
    #print("c: exiting run_consumer()")
    result_thread.put(out_res)



if __name__ == "__main__":
    run_consumer()



