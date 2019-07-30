#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:11:09 2019

@author: angelmanzur
"""
import json
import numpy as np
print('Reading full reicpes')
data_file = '../data/Merged_Recipe.json'
with open(data_file, 'r') as file:
    new_data = json.load(file)
    
    
num_ingredients = []
num_instructions = []
for recipe in new_data:
    n_ings = len(recipe['ingredients'])
    n_instr = len(recipe['instructions'])
    num_ingredients.append(n_ings)
    num_instructions.append(n_instr)
    
    
import matplotlib.pyplot as plt

n_ings = np.array(num_ingredients)
n_instr= np.array(num_instructions)

print('Intructions')
print('Avg Number of instructions {:1.0f}'.format(n_instr.mean()))
print('Min, Max instructions: {}, {}'.format(n_instr.min(), n_instr.max()))

print('Ingredients')
print('Avg Number of ingredients {:1.0f}'.format(n_ings.mean()))
print('Min, Max ingredientss: {}, {}'.format(n_ings.min(), n_ings.max()))




plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()


fig = plt.figure(figsize=(8,6))
plt.hist(n_ings, bins=10);
plt.title('Ingredients', fontsize=32, color='darkred')
plt.xlabel('Number of ingredients', fontsize=24)
plt.ylabel('', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.savefig('figs/Ingredient_hist.png')


fig = plt.figure(figsize=(8,6))
plt.hist(n_instr, bins=20);
plt.title('Instructions', fontsize=32, color='darkred')
plt.xlabel('Number of instructions', fontsize=24)
plt.ylabel('', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.savefig('figs/Instructions_hist.png')



fig = plt.figure(figsize=(8,6))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

ax1.hist(n_ings, bins=20);
ax1.set_xlabel('Number of Ingredients', fontsize=24)
ax1.tick_params(labelsize=16)
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)
ax2.hist(n_instr, bins=20);
ax2.set_xlabel('Number of Instructions', fontsize=24)
ax2.tick_params(labelsize=16)

plt.savefig('figs/Ingredients_Instructions_hist.png')
#plt.xticks(fontsize=20)
#plt.yticks(fontsize=20)


fig = plt.figure(figsize=(8,6))
plt.hist2d(n_ings, n_instr,bins=40)
