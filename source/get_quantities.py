#!/usr/bin/env python3

import re
import json
from extract_desserts import *
import numpy as np
from pattern.text.en import singularize
# open the dessert ingredients file


raw_ingrs = get_raw_ingredients()
raw_recipe = get_raw_data()


item = np.random.randint(len(raw_recipe))
# item = 1142
print('item',item)
# get a list of the ingredients from the raw recipe
recipe_ingredients_wq = raw_recipe[item]['ingredients']
# get a list ot ingredient without quantities
ingredient_list = raw_ingrs[item]['ingredients']

raw_ingrs[item]['qty'] = []
amounts = find_amounts(recipe_ingredients_wq,ingredient_list)
for amt in amounts:
    raw_ingrs[item]['qty'].append({'qty':amt})

def find_amounts(recipe_ingredients_wq,ingredient_list):
    n_ingredients = len(recipe_ingredients_wq)
    pattern = r'[0-9]?\s?[0-9]+'+re.escape('/')+r'?[0-9]?\s+\w+'
    total_amount = 0
    amounts = []
    for i in range(n_ingredients):  
        full_ingredient = recipe_ingredients_wq[i]['text']
        m = re.findall(pattern, full_ingredient)
        amount = 0 
    
        unit = 'ml'
        if len(m)>0:
            print(full_ingredient,'\n \t \t',m,'\n')
            entries = m[0].split(' ')
            if len(entries)==2:
                if entries[0].isnumeric():
                    amount = int(entries[0])*convert_to_ml(entries[1])
            elif len(entries)==3:
                if entries[0].isnumeric() and entries[1].isnumeric():
                    amount = int(entries[0]) * int(entries[1])*convert_to_ml(entries[2])
            amounts.append(amount)
        
            print('\t \t amount {:1.2f} ml'.format(amount))

    amounts = np.array(amounts)/np.sum(amounts)
    print(amounts)
    # add quantity
    return amounts







def convert_to_ml(unit):
    unit = singularize(unit)
    conversion = {'cup': 236.588,
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
    qty_ml = 0
    try:
        qty_ml = conversion[unit]
    except:
        qty_ml = 1

    return qty_ml

        
