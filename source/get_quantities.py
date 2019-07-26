#!/usr/bin/env python3

import re
import json
from extract_desserts import *
import numpy as np
from pattern.text.en import singularize
# open the dessert ingredients file




def get_all_quantities(raw_data, raw_ingredients):
    conv_dict = get_conversion_dictionary()
    for item in range(len(raw_data)):
       # print('item {}'.format(item))
        # get a list of the ingredients
     #   item = 19995
        recipe_ingredients_wq = raw_data[item]['ingredients']
        # get a list ot ingredient without quantities
        ingredient_list = raw_ingredients[item]['ingredients']
        # set an empty list for the amounts    
        raw_ingredients[item]['qty'] = []
        amounts = find_amounts(recipe_ingredients_wq,ingredient_list,conv_dict)
        #print(amounts)
        if len(amounts) != len(ingredient_list):
            print('problem with item',item)
            print('\t {}. {}'.format(len(amounts),len(ingredient_list)))
        for amt in amounts:
            raw_ingredients[item]['qty'].append({'qty':amt})

    return raw_ingredients




def find_amounts(recipe_ingredients_wq,ingredient_list,conv_dict):
    n_ingredients = len(recipe_ingredients_wq)
    pattern = r'[0-9]?-?[0-9]?\s?[0-9]+.?'+re.escape('/')+r'?[0-9]?\s+\w+'
    if n_ingredients==1:
         return np.ones(n_ingredients)
    
    amounts = np.zeros(n_ingredients)
    for i in range(n_ingredients):  
        full_ingredient = recipe_ingredients_wq[i]['text']
        reg_ingredients = re.findall(pattern, full_ingredient)
        amount = 0 
        
        # not exactly since some units are weight, some are volume
        # unit = 'ml' 
        if len(reg_ingredients)>0:
            # if there are more than one quantities, ( 1 pkg (7.5 oz)) select one
            if len(reg_ingredients)>1:
                one_ingredient = get_best_unit(reg_ingredients, conv_dict)
            else:
                one_ingredient = reg_ingredients[0] 
                    
            #print(full_ingredient,'\n \t \t',one_ingredient, end='\t')
            
            entries = one_ingredient.split(' ')
            #print('\n -> ',entries)
            try:
                if len(entries)==2:
                    
                    tpattern = '[0-9]+-[0-9]+\.[0-9]+'
                    if re.match(tpattern, entries[0]):
                        numbers = entries[0].split('-')
                        amount = float(numbers[1])
                    elif entries[0].strip('-').isnumeric() or r'.' in entries[0]:
                        amount = float(entries[0].strip('-'))*convert_to_ml(entries[1],conv_dict)
                    elif r'/' in entries[0] and r'-' not in entries[0]:
                        fraction = entries[0].split(r'/')
                        if len(fraction)==2:
                            amount = int(fraction[0])/int(fraction[1])*convert_to_ml(entries[1],conv_dict)
                    elif r'/' in entries[0] and r'-' in entries[0]:
    
                        number = entries[0].split(r'-')
                        fraction = number[1].split(r'/')
                        # print(number, fraction)
                        if len(fraction)==2 and number[0].isnumeric():
                            amount = (float(number[0]) + int(fraction[0])/int(fraction[1]))*convert_to_ml(entries[1],conv_dict)
                        else:
                            amount = ( int(fraction[0])/int(fraction[1]))*convert_to_ml(entries[1],conv_dict)
                elif len(entries)==3:
                    if entries[0].isnumeric() and entries[1].isnumeric():
                        amount = int(entries[0]) * int(entries[1])*convert_to_ml(entries[2],conv_dict)
                    elif entries[0].isnumeric() and r'/' in entries[1]:
                        number = int(entries[0])
                        fraction = entries[1].split(r'/')
                        if len(fraction)==2:
                            amount = (number + int(fraction[0])/int(fraction[1]))*convert_to_ml(entries[2], conv_dict)
            except:
                amount = 0
                
            amounts[i] = np.abs(amount)
            #amounts.append(np.abs(amount))    
            #print(' amount {:1.2f} ml'.format(amount))
    
    qts = np.array(amounts)/np.sum(amounts)
    return qts


def get_conversion_dictionary():
    """
    define a dictionary with conversion units into ml
    """
    conversion = {'cup': 236.588,
                'c': 236.588,
                'can': 236.588,
                'small': 118.294,
                'medium': 236.588,
                'onion' : 236.588,
                'red' : 236.588,
                'yellow' : 236.588,
                'green' : 236.588,
                'baby': 236.588,
                'bell' : 236.588,
                'large' : 354.882,
                'oz': 29.5735,
                'ounce': 29.5735,
                'teaspoon': 4.92892,
                'tsp': 4.92892, 
                't':   4.92892,
                'ts':  4.92892,
                'tspn':4.92892,
                'garlic': 4.92892,
                'clove':4.92892,
                'pinch': 0.616115,
                'dash': 0.616115,
                'tablespoon': 14.7868,
                'tbsp': 14.7868,
                #'T': 14.7868,
                'tbls': 14.7868,
                'tb': 14.7868,
                'egg': 51.75368,
                'slice': 22.18015,
                'lb':  453.59,
                'pound': 453.59,
                'pint': 473.176,
                'kg': 1000.0,
                'head': 906.
    }
    return conversion


def get_best_unit(ingred_list, conversion):
    """
    Some ingredients, have different options like
      1 pkg (7.5 oz) 
    the code will identify both, this function tells which one to use
    """
    for ingred in ingred_list:
        # print(ingred)
        inspect = ingred.strip().split(' ')
        # print(inspect[1].lower())

        new_unit = singularize(inspect[1].lower())

        if new_unit in list(conversion.keys()):
            return ingred
    
    # could not find it in the list use the first one
    return ingred_list[0]


def convert_to_ml(unit, conversion):
    """
    Convert the unit, usually in volume, 
    to a weight so we can calculate the fraction of ingredient present/
    """
    unit = singularize(unit.lower())

    qty_ml = 0
    try:
        qty_ml = conversion[unit]
    except:
        qty_ml = 1

    return qty_ml



        
#raw_data = get_raw_data('sample_layer1.json')
#raw_ingredients = get_raw_ingredients('sample_det_ingrs.json')

#item = np.random.randint(len(raw_data))
# item = 15881
#print('item',item)
# get a list of the ingredients from the raw recipe
#raw_ingredients = get_all_quantities(raw_data, raw_ingredients)
