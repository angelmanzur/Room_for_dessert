#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:57:40 2019

@author: angelmanzur
"""

import json
import numpy as np
#to simplufy the analysis, let's break the data into several files, 50k recipes each
def get_data_slice(layer_data, nfile, istart, iend, ingredient=False):
       
    sample_layer_data = []
    for i in range(istart, iend):
        #print('-',end='')
        sample_layer_data.append(layer_data[i])
        if i%5000==0:
            print('{}/{} done'.format(i-istart,iend-istart))
    if ingredient:
        save_file = '../data/ingredients/sample_det_ingrs_{}.json'.format(nfile)
    else:
        save_file = '../data/full_recipes/sample_layer1_{}.json'.format(nfile)
            
        
    with open(save_file, 'w') as outfile:
        json.dump(sample_layer_data, outfile)

    print('\n created file',save_file)  
    pass



def break_data_files():
    """ 
    break the original json file (1,029,720 entries), into files containing
        50,000 entries each
    """
    #total number of recipes
    print('Reading full reicpes')
    data_file = '../data/layer1.json'
    with open(data_file, 'r') as file:
        layer_data = json.load(file)
    total_recipes = len(layer_data)
    
    print('Reading extracted ingredients')
    rdata_file = '../data/det_ingrs.json'
    with open(rdata_file, 'r') as file:
        ingr_data = json.load(file)
        
        
    batch_size = 25000
    
    nfiles = int(total_recipes/batch_size)+1
    print(nfiles)
    first_recipe = 0
    for i in range(nfiles):
        start = i*batch_size
        end = min((i+1)*batch_size,total_recipes)
        
        print('File {} contains entries {} to {}'.format(i,start, end  ))
        get_data_slice(layer_data, i, start, end,ingredient=False)
        get_data_slice(ingr_data, i, start, end,ingredient=True)
        
    pass










    