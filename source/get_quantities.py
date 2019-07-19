#!/usr/bin/env python3

import re
import json
from extract_desserts import *
import numpy as np
from pattern.text.en import singularize
# open the dessert ingredients file


raw_ingrs = get_raw_ingredients()
raw_recipe = get_raw_data()
conv_dict = get_conversion_dictionary()

item = np.random.randint(len(raw_recipe))
item = 15881
print('item',item)
# get a list of the ingredients from the raw recipe
recipe_ingredients_wq = raw_recipe[item]['ingredients']
# get a list ot ingredient without quantities
ingredient_list = raw_ingrs[item]['ingredients']

raw_ingrs[item]['qty'] = []
amounts = find_amounts(recipe_ingredients_wq,ingredient_list,conv_dict)
print(amounts)
for amt in amounts:
    raw_ingrs[item]['qty'].append({'qty':amt})

def find_amounts(recipe_ingredients_wq,ingredient_list,conv_dict):
    n_ingredients = len(recipe_ingredients_wq)
    pattern = r'[0-9]?\s?[0-9]+.?'+re.escape('/')+r'?[0-9]?\s+\w+'
    total_amount = 0
    amounts = []
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
                    
            print(full_ingredient,'\n \t \t',one_ingredient,'\n')
            
            entries = one_ingredient.split(' ')
            if len(entries)==2:
                if entries[0].isnumeric():
                    amount = int(entries[0])*convert_to_ml(entries[1],conv_dict)
                elif r'/' in entries[0]:
                    number = entries[0].split(r'/')

                    if len(number)==2:
                        amount = int(number[0])/int(number[1])*convert_to_ml(entries[1],conv_dict)
                                         
            elif len(entries)==3:
                if entries[0].isnumeric() and entries[1].isnumeric():
                    amount = int(entries[0]) * int(entries[1])*convert_to_ml(entries[2],conv_dict)
            amounts.append(amount)
        
            print('\t \t amount {:1.2f} ml'.format(amount))

    qts = np.array(amounts)/np.sum(amounts)
    # print(qts)
    # add quantity
    return qts


def get_conversion_dictionary():
    """
    define a dictionary with conversion units into ml
    """
    conversion = {'cup': 236.588,
                'oz': 29.5735,
                'ounce': 29.5735,
                'teaspoon': 4.92892,
                'tsp': 4.92892, 
                't':   4.92892,
                'ts':  4.92892,
                'tspn':4.92892,
                'tablespoon': 14.7868,
                'tbsp': 14.7868,
                #'T': 14.7868,
                'tbls': 14.7868,
                'tb': 14.7868,
                'egg': 51.75368,
                'slice': 22.18015,
                'lb':  453.59,
                'pint': 473.176,
                'kg': 1000.

    }
    return conversion


def get_best_unit(ingred_list, conversion):
    """
    Some ingredients, have different options like
      1 pkg (7.5 oz) 
    the code will identify both, this function tells which one to use
    """
    for ingred in ingred_list:
        inspect = ingred.split(' ')
        print(inspect)
        new_unit = singularize(inspect[1].lower)

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

        
