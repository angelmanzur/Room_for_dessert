#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:08:18 2019

@author: angelmanzur
"""
import os, os.path
import json
from extract_desserts import *
from get_quantities import *


def clean_main():
    """
    Get the raw data and raw ingredients, extract the desserts, clean the dataset, 
    and save them all as a list
    """
    #Full recipe files
    full_data_dir = "../data/full_recipes"
    
    full_file_names = os.listdir(full_data_dir)
    print(len(full_file_names))

    
    ingr_data_dir = "../data/ingredients"
    
    ingr_file_names = os.listdir(ingr_data_dir)
    print(len(ingr_file_names))
    
    nfiles = len(ingr_file_names)
    total_recipes = 0
    dessert_recipes = 0   
    total_clean_desserts = 0
    total_classified_desserts = 0
    
    new_data = []
    new_data_ings = []
    new_ingredients = []
    target = []
#    for i in range(nfiles):
    for i in range(5):
        #full data
        fullname = 'full_recipes/sample_layer1_{}.json'.format(i)
        print(fullname)
        
        #ingr dta
        ingrname = 'ingredients/sample_det_ingrs_{}.json'.format(i)
        print(ingrname)
        
        
        raw_data = get_raw_data(fullname)
        raw_ingredients = get_raw_ingredients(ingrname)
        
        desserts, dessert_ings = find_desserts(raw_data, raw_ingredients)
        
        total_recipes += len(raw_data)
        dessert_recipes += len(desserts)
        del raw_data, raw_ingredients #clear some memory
        print('Will look at {} dessert recipes, out of {} (~{:1.1f}%)'.format(
                            dessert_recipes, total_recipes,
                            dessert_recipes/total_recipes*100))
        
        #get the amounts for each dessert
        dessert_ings_wq = get_all_quantities(desserts, dessert_ings)
        clean_ingredients =  clean_dessert_ingredients(dessert_ings_wq)
        
        total_clean_desserts += len(clean_ingredients)
        print(total_clean_desserts)
        
        
        tnew_data, tnew_data_ings, tnew_ingredients, ttarget = classify_desserts(desserts, clean_ingredients)
        new_data += tnew_data
        new_data_ings += tnew_data_ings
        new_ingredients += tnew_ingredients
        target += ttarget
        
        total_classified_desserts += len(new_data)
        
        print('classified desserts: ',total_classified_desserts)
        
    ### save the lists
    save_file = '../data/Merged_Recipe.json'
    with open(save_file, 'w') as outfile:
        json.dump(new_data, outfile)
        
    save_file = '../data/Merged_Ingredients_only.json'
    with open(save_file, 'w') as outfile:
        json.dump(new_data_ings, outfile)
        
    save_file = '../data/Merged_Recipe_Ingredients.json'
    with open(save_file, 'w') as outfile:
        json.dump(new_ingredients, outfile)
                  
    save_file = '../data/Merged_Target.json'
    with open(save_file, 'w') as outfile:
        json.dump(target, outfile)
    pass


if __name__ == "__main__":
    clean_main()