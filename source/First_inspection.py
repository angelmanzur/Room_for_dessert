#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 20:57:34 2019

@author: angelmanzur
"""

from extract_desserts import *
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
#load the raw data
# lets use the default value that only loads 250 recipes,
# instead of 1 million
raw_data = get_raw_data()
raw_ingredients = get_raw_ingredients()

print('N recipes: ',len(raw_data),'\t N list of ingredients ', len(raw_ingredients))

desserts, dessert_ings = find_desserts(raw_data, raw_ingredients)

total_recipes = len(raw_data)
dessert_recipes = len(desserts)
dessert_ingredients = len(dessert_ings)
print('Will look at {} dessert recipes, out of {} (~{:1.1f}%)'.format(dessert_recipes, total_recipes,dessert_recipes/total_recipes*100))

all_ingredients = []

for recipe_ings in dessert_ings:
    n_ingredients = len(recipe_ings['valid'])
    for item, ingredient in enumerate(recipe_ings['ingredients']):
        if recipe_ings['valid'][item] ==True:
            all_ingredients.append(ingredient['text'])
            

stop_words =set(stopwords.words("english"))


ingredient_set = set(all_ingredients)

ps = PorterStemmer()
stemmed_words=[]


    
print('with {} recipes, there are {} ingredients, and {} unique ingredients.'.format(
                dessert_recipes, len(all_ingredients), len(ingredient_set)))

fdist = FreqDist(all_ingredients)
print(fdist)

fdist.most_common(10)



fig = plt.figure(figsize=(8, 6))
#plt.ylim(0,120)
fdist.plot(10,cumulative=False)

for w in ingredient_set:
    stemmed_words.append(ps.stem(w))
    
fdist = FreqDist(stemmed_words)
print(fdist)

fdist.most_common(10)



fig = plt.figure(figsize=(8, 6))
#plt.ylim(0,120)
fdist.plot(10,cumulative=False)