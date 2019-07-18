#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:57:40 2019

@author: angelmanzur
"""

import json

#look at all the data
def get_layer1():
    data_file = '../data/layer1.json'
    with open(data_file, 'r') as file:
        layer_data = json.load(file)
        
    sample_layer_data = []
    for i in range(20000):
        print('-',end='')
        sample_layer_data.append(layer_data[i])
        if i%50==0:
            print(i,'/ 20000 done')
            
    save_file = '../data/sample_layer1.json'
    
    with open(save_file, 'w') as outfile:
        json.dump(sample_layer_data, outfile)

    print('\n got the downsize file')  
    pass

def get_ings():
    data_file = '../data/det_ingrs.json'
    with open(data_file, 'r') as file:
        layer_data = json.load(file)
        
    sample_layer_data = []
    for i in range(20000):
        print('-',end='')
        sample_layer_data.append(layer_data[i])
        if i%50==0:
            print(i,'/ 20000 done')
            
    save_file = '../data/sample_det_ingrs.json'
    
    with open(save_file, 'w') as outfile:
        json.dump(sample_layer_data, outfile)

    print('\n got the downsize file')  
    pass