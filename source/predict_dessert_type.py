#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:50:57 2019

@author: angelmanzur
"""

from extract_desserts import *
from get_quantities import *
from pattern.text.en import singularize
from nltk import FreqDist
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed=2019)

import re
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    datefmt='%H:%M:%S',
                   level=logging.INFO)


import warnings
warnings.filterwarnings('ignore')

#load the raw data

# print the categories
categories = ['cake', 'cookies', 'pie','bread','cupcake','candy', 'pudding','custard']
categories = ['cake', 'cookies', 'pie', 'pudding']
#=============================================================================#
#classify the desserts
def classify_desserts(recipes, recipe_ingredients):
    """
    get the recipes and the ingredients for the recipes, and based on the title, 
    classify the recipes as one of the items in categories
    If the function can't determine the recipe, it will be ignored
    
    Output:
        recipes_list, ingredients_list, categories for recipes_list
    """
    n_recipes = 0
    n_classified =0
    new_recipe_list = []
    new_recipe_ingredients = []
    category_list = []
    for item, recipe in enumerate(recipes):
        title = recipe['title']
        title = title.lower().split(' ')
        for category in categories:
            if category in title:
#                 print(category)
                n_classified +=1
                recipe['type'] = category
                new_recipe_list.append(recipe)
                all_ingredients = ''
                for ingredient in recipe_ingredients[item]['ingredients']:
                    all_ingredients += ingredient['text'] + ' '
                new_recipe_ingredients.append(all_ingredients)
                category_list.append(categories.index(category))
                break
                  
        n_recipes += 1
        if n_recipes%1000==0:
            logging.info("read {0} recipes".format(n_recipes))
#             break
    return new_recipe_list, new_recipe_ingredients, category_list




#=============================================================================#
def sequence_data(text, ndim=100):
    """
    Get the word index and the pad_sequences from keras
    """
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    import spacy
    
    max_num_words= 1000
    max_seq_length = 100
    tokenizer = Tokenizer(num_words = max_num_words)    

    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index    

    data = pad_sequences(sequences, maxlen=max_seq_length)

    return data, word_index


#=============================================================================#
def train_val_split(data, label, vsplit=0.20):
    """
    Split the data into tain and validation test sets. 
    the default validation split it 20%
    NOTE, do not pass the test test
    
    Usage:
        c_train, y_train, x_val, y_val = train_val_split(data, target, vsplit=0.20)
    """
    validation_split = vsplit
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    label = label[indices]
    
    nb_validation_samples = int(validation_split*data.shape[0])
    x_train = data[:-nb_validation_samples]
    y_train = label[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = label[-nb_validation_samples:]
    
    return x_train, x_val, y_train, y_val

#=============================================================================#
def get_glove_embeddings(ndim=50):
    """
    Get the gloVe embedings, and return the embedding index.
    Parameter:
        ndim:  dimension of the vector, options: 50 (defulat), 
                                                100, 200, or 300
    """
    embedding_dim = ndim
    embedding_index = {}
    f = open('../data/glove_data/glove.6B.{}d.txt'.format(embedding_dim))
    for line in f:
        values = line.split()
        word = values[0]
    
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close
    
    print('found word vecstors: ', len(embedding_index))
    return embedding_index
#=======================================================================#
 


not_a_flavor_list = ['flour','cake mix','baking soda','baking powder', 'canola oil',
                     'vegetable oil','cornstarch','shortening','margarine','yeast',
                     'gelatin' ,'coloring','corn syrup','cooking spray', 
         'crisco', 'cake', 'crisco','xanthan','bicarbonate','pie shell','pie crust',
                     'cornmeal','splenda','stevium','ice cube', 'gluten','bread'
                    ]

generic_ingredient_list = ['pumpkin', 'vanilla','cinnammon','cocoa powder',
                           'chocolate chip','salted butter', 'graham cracker','mint',
                          'salted butter','coffee']

def in_not_a_flavor_list(ingredient):
    for word in not_a_flavor_list:
        if re.search(word, ingredient):
            return True   
    return False


def is_generic_ingredient(ingredient):
    for word in generic_ingredient_list:
#         print(word,ingredient)
        if re.search(word, ingredient):
            return True, word
    return False, ingredient

def clean_dessert_ingredients(all_ingredients):
    count =0
    for item, ingredients_list in enumerate(all_ingredients):
        to_remove = []
        for ingr_item, ingredient in enumerate(ingredients_list['ingredients']):
            tmp_ingredient = ingredient['text'].lower().replace(' - ','-')
            tmp_ingredient = tmp_ingredient.replace("'",'')
            tmp_ingredient = tmp_ingredient.replace(" & ",'&')
            tmp_ingredient = tmp_ingredient.replace('fat free', 'fat-free')
#             tmp_ingredient = singularize(tmp_ingredient)
            
#             print(ingredient['text'].replace(' - ','-'))
            if re.search('water', tmp_ingredient) and not re.search('watermelon',tmp_ingredient) and not re.search('rose water',tmp_ingredient):
                to_remove.append(ingr_item)
            
            if in_not_a_flavor_list(tmp_ingredient):
                to_remove.append(ingr_item)
            elif re.match('oil', tmp_ingredient):
                to_remove.append(ingr_item)

                        
            
            if re.search('heavy', tmp_ingredient) and re.search('cream',tmp_ingredient):
                tmp_ingredient='heavy cream'
            
            generic_ingr, ingr = is_generic_ingredient(tmp_ingredient)
            if generic_ingr:
                tmp_ingredient = ingr

                
#             elif (re.search('purpose flour', tmp_ingredient) 
#                   or re.search('cake flour',tmp_ingredient) 
#                   or re.search('rising flour', tmp_ingredient) 
#                   or re.search('plain flour', tmp_ingredient)
#                   or re.search('white flour', tmp_ingredient)
#                   or re.search('raising flour', tmp_ingredient)

#                  ):
#                 tmp_ingredient = 'flour' 
            elif re.search('brown sugar', tmp_ingredient) or re.search('demerara sugar', tmp_ingredient):
                tmp_ingredient = 'brown sugar'
            
            elif re.search('powdered sugar', tmp_ingredient) or re.search('confectioners sugar',tmp_ingredient) :
                tmp_ingredient ='powdered sugar'
            elif re.search('sugar', tmp_ingredient) or re.search('white sugar', tmp_ingredient) or re.search('granulated sugar',tmp_ingredient):
                tmp_ingredient = 'sugar'

            elif re.search('semi-sweet', tmp_ingredient) and re.search('chocolate',tmp_ingredient):
                tmp_ingredient = 'chocolate'

            if (not re.search('flour', tmp_ingredient) 
                and not re.search('molasses',tmp_ingredient)
                and not re.search('oats',tmp_ingredient)):
                tmp_ingredient = singularize(tmp_ingredient)


            all_ingredients[item]['ingredients'][ingr_item]['text'] = tmp_ingredient
            
        if len(to_remove)>0:
            
            to_remove.sort(reverse=True)
            
                
#             print(item,'to remove', to_remove)
            for i in to_remove:
#                 print('try to remove', all_ingredients[item]['ingredients'][i])
                del all_ingredients[item]['ingredients'][i]
#         for iremove in to_remove:
#             del all_ingredients
#         print('--------------------------')
        if item%500==0:
            logging.info("read {0} recipes".format(item))
    return all_ingredients

def print_dessert_ingredients(all_ingredients):
    count =0
    for item, ingredients_list in enumerate(all_ingredients):
        for ingr_item, ingredient in enumerate(ingredients_list['ingredients']):
            
            
            print(ingredient['text'])
        
        print('--------------------------')
        if item==50:
            break
    return all_ingredients


#=========================================================================#   
#load the data
raw_data = get_raw_data('sample_layer1.json')
raw_ingredients = get_raw_ingredients('sample_det_ingrs.json')

#get the desserts
desserts, dessert_ings = find_desserts(raw_data, raw_ingredients)#


total_recipes = len(raw_data)
dessert_recipes = len(desserts)
dessert_ingredients = len(dessert_ings)
print('Will look at {} dessert recipes, out of {} (~{:1.1f}%)'.format(
                            dessert_recipes, total_recipes,
                            dessert_recipes/total_recipes*100))


clean_ingredients =  clean_dessert_ingredients(dessert_ings);

#classify the recipes
#new_data, new_data_ings, target = classify_desserts(desserts,dessert_ings)
new_data, new_data_ings, target = classify_desserts(desserts, clean_ingredients)
print('classified {0}/{1}'.format(len(new_data),len(new_data_ings)))

# set the data into a pandas dataframe
import pandas as pd
df = pd.DataFrame([new_data_ings,target]).T
df.columns = ['ingredients','dessert']
#df['dessert'] = np.where(df['dessert']=='cake', 1, 0)

# get the test and the labels
text = df.ingredients.values
label = df.dessert.values
from keras.utils import to_categorical

# set tha label to categorical
data, word_index = sequence_data(text)
                                 
label = to_categorical(np.asarray(label))
# break into train, validation (20%)

# get train val data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, label)
x_train, x_val, y_train, y_val = train_val_split(X_train, y_train)


# get the glove data, 


# create an empty matrix based on gloVe embeddings
embedding_dim = 50
embedding_index = get_glove_embeddings(embedding_dim)
embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))
#embedding_matrix.shape

# using the gloVe embeddings
for word,i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
from keras.layers import Embedding
embedding_layer = Embedding(len(word_index)+1,
                            embedding_dim,weights=[embedding_matrix],
                            input_length=max_seq_length,
                            trainable=False)

from keras.layers import Bidirectional,GlobalMaxPool1D,Conv1D
from keras.layers import LSTM,Input,Dense,Dropout,Activation
from keras.models import Model

inp = Input(shape=(max_seq_length,))
x = embedding_layer(inp)
x = Bidirectional(LSTM(embedding_dim,return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(embedding_dim,activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(len(categories),activation='sigmoid')(x)
model = Model(inputs=inp,outputs=x)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=40,batch_size=128);
score = model.evaluate(x_val,y_val)
print('score: {}%'.format(score[1]*100,))
y_pred_test = model.predict(X_test)

score_test = model.evaluate(X_test, y_test)
print('test score evaluation {}%'.format(score_test[1]*100))




#=============================================================================#
#using the word2vec as embedding

ingredients_per_recipe = []
all_ingredients = []
        
count =0
for drecipe in clean_ingredients:
    ings =[]
    for entry in drecipe['ingredients']:
        ings.append(entry['text'])
        all_ingredients.append(entry['text'])
    ingredients_per_recipe.append(ings)
    count += 1
    if count%500==0:
        logging.info("read {0} ingredients".format(count))
        
vec_size = embedding_dim
model = gensim.models.Word2Vec(
    size=vec_size,
    window=5,
    min_count=1,
    workers=10,
    alpha=0.02,
    iter=4,
    sg=0)

model.build_vocab(ingredients_per_recipe, progress_per=1000)
model.train(ingredients_per_recipe, total_examples=model.corpus_count,
           epochs=40, report_delay=1)
model.init_sims(replace=True)


w1='pumpkin'
# w1='rum'
model.wv.most_similar(positive=w1, topn=10)
unique_ingredients = set(all_ingredients)

embedding_matrix_w2v = np.zeros((len(unique_ingredients), embedding_dim))
for i,word in enumerate(unique_ingredients):
    embedding_vector = model.wv.__getitem__(word)
#    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix_w2v[i] = embedding_vector
        
 embedding_layer = Embedding(len(unique_ingredients),
                            embedding_dim,weights=[embedding_matrix_w2v],
                            input_length=max_seq_length,
                            trainable=False)       

inp = Input(shape=(max_seq_length,))
x = embedding_layer(inp)
x = Bidirectional(LSTM(embedding_dim,return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(embedding_dim,activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(len(categories),activation='sigmoid')(x)
model = Model(inputs=inp,outputs=x)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=40,batch_size=128);
score = model.evaluate(x_val,y_val)
print('score: {}%'.format(score[1]*100,))
y_pred_test = model.predict(X_test)

score_test = model.evaluate(X_test, y_test)
print('test score evaluation {}%'.format(score_test[1]*100))        
### count Vectorizer
#============================#
from sklearn.feature_extraction.text import CountVectorizer
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocabulary
vectorizer.fit(text)

#print(vectorizer.vocabulary_)
#encode document
vector = vectorizer.transform(text)





## Word Frequencies with TfidVectorizer

## Word Hashing 