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
import pickle

import json
import re
import gensim
import pandas as pd

#some keras packages
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Bidirectional,GlobalMaxPool1D
from keras.layers import LSTM,Input,Dense,Dropout
from keras.models import Model

# some sklearn packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#load the raw data
max_num_words= 1000
max_seq_length = 100
# print the categories

#=============================================================================#
def sequence_data_w2v(text, ndim=50):
    """
    Get the word index and the pad_sequences from keras
    """
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    
    tokenizer = Tokenizer(num_words = max_num_words)    

    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index    

    data = pad_sequences(sequences, maxlen=max_seq_length)#,padding='post')

    return data, word_index


def sequence_data(text, ndim=50):
    """
    Get the word index and the pad_sequences from keras
    """
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences

    
    tokenizer = Tokenizer(num_words = max_num_words)    

    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index    

    data = pad_sequences(sequences, maxlen=max_seq_length)#,padding='post')

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
 
# these ingredients are not a flavor, so let's not include them 
not_a_flavor_list = ['flour','cake mix','baking soda','baking powder', 
                     'canola oil','vegetable oil','cornstarch','shortening',
                     'margarine','yeast','gelatin' ,'coloring',
                     'corn syrup','cooking spray', 'crisco', 'cake', 
                     'crisco','xanthan','bicarbonate','pie shell',
                     'pie crust','cornmeal','splenda','stevium',
                     'ice cube', 'gluten','bread'
                    ]

generic_ingredient_list = ['pumpkin', 'vanilla','cinnammon','cocoa powder',
                           'chocolate chip','salted butter', 'graham cracker','mint',
                          'salted butter','coffee']
#=========================================================================# 
def in_not_a_flavor_list(ingredient):
    """
    Return true if ingredient is in not_a_flavor_list 
    Output:
        Boolean
    """
    
    for word in not_a_flavor_list:
        if re.search(word, ingredient):
            return True   
    return False

#=========================================================================# 
def is_generic_ingredient(ingredient):
    """
    Find if ingreditn is in the generic_ingredient_list
    Output:
        Boolean
    """
    for word in generic_ingredient_list:
#         print(word,ingredient)
        if re.search(word, ingredient):
            return True, word
    return False, ingredient
#=========================================================================# 
def clean_dessert_ingredients(all_ingredients):
    """
    Clean the ingredients, following several rules:
        lowercase the words, singularize, strip extra spaces. 
        And combine ingredeints, for examples light brown sugar and demerara sugar -> brown sugar
        If an ingredient is not a flavor, it is set to False in the list.
        
    Input:
        ingredient list
    Output:
        ingredient list after cleaning the ingredients. 
    """
    count =0
    for item, ingredients_list in enumerate(all_ingredients):
        to_remove = []
        for ingr_item, ingredient in enumerate(ingredients_list['ingredients']):
            tmp_ingredient = ingredient['text'].lower().replace(' - ','-')
            tmp_ingredient = tmp_ingredient.replace("'",'')
            tmp_ingredient = tmp_ingredient.replace(" & ",'&')
            tmp_ingredient = tmp_ingredient.replace('fat free', 'fat-free')

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
                           
            try:
                for i in to_remove:
                    all_ingredients[item]['valid'][i] = False
                    
            except:
                print(to_remove, all_ingredients[item])
                input()

        if item%1000==0:
            logging.info("read {0} recipes".format(item))
    return all_ingredients

#=========================================================================# 
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
def get_glove_embedding_matrix(word_index, dimension):
    """
    Get the embedding matrix using gloVe
    Input:
        word index, dimension = (50,100,200,300)
    Output:
        embedding matrix
    """
    embedding_dim = dimension
   
    embedding_index = get_glove_embeddings(embedding_dim)
 
    embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))    

    for word,i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
#=========================================================================# 
def get_random_embedding_matrix(word_index, dimension):
    """
    Get the embedding matrix using random numbers
    Input:
        word index, dimension = (50,100,200,300)
    Output:
        embedding matrix
    """
    embedding_dim = dimension
   
 #   embedding_index = get_glove_embeddings(embedding_dim)
 
    embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))    

    for word,i in word_index.items():
        embedding_vector = np.random.rand(dimension)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix  
 
#=========================================================================# 
def get_w2v_embedding_matrix(word_index, dimension, w2v_model):
    """
    Get the embedding matrix using the word2vec model passed
    Input:
        word index, dimension = (50,100,200,300), word2vec model
    Output:
        embedding matrix
    """
    embedding_dim = dimension

    embedding_matrix_w2v = np.zeros((len(word_index), embedding_dim))
    for i,word in enumerate(word_index):
        try:
            embedding_vector = w2v_model.wv.__getitem__(word)
    #       embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix_w2v[i] = embedding_vector
        except:
            pass
    return embedding_matrix_w2v

#=========================================================================# 
def embedding_LSTM(data, label, embedding_matrix, word_index, dimension, extra_dim, trainable=False):    
    """
    Run an LSTM model on the data with target: label, using passed embedding_matrix
    Input:
        data, label, embedding_matrix, word_index, dimension, extra_dim=0,1 if 
        word_index a list, or a dictionary)
    Output:
        the Rnn model
    """
    X_train, X_test, y_train, y_test = train_test_split(data, label)
    x_train, x_val, y_train, y_val = train_val_split(X_train, y_train)
    
    word_index = word_index

    # using the gloVe embeddings

    embedding_dim = dimension
    embedding_layer = Embedding(len(word_index)+extra_dim,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_seq_length,
                            trainable=trainable)


    inp = Input(shape=(max_seq_length,))
    x = embedding_layer(inp)
    x = Bidirectional(LSTM(embedding_dim,return_sequences=True,dropout=0.1,recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(embedding_dim,activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(len(categories),activation='sigmoid')(x)
    rnn_model = Model(inputs=inp,outputs=x)

    rnn_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    rnn_model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=20,batch_size=128);
    score = rnn_model.evaluate(x_val,y_val)
    print('score: {}%'.format(score[1]*100,))
    y_pred_test = rnn_model.predict(X_test)

    score_test = rnn_model.evaluate(X_test, y_test)
    print('test score evaluation {}%'.format(score_test[1]*100))

    return rnn_model, x_train, y_train, x_val, y_val, X_test, y_test
#=========================================================================#  
def get_Word2Vec(dimension, ingredients_per_recipe): 
    """
    Fit a word2vec model based on the documents passed.
    /input:
        List of documents
    Output:
        word2vec model
    """
    vec_size = dimension
    w2v_model = gensim.models.Word2Vec(
            size=vec_size,
            window=5,
            min_count=1,
            workers=10,
            alpha=0.02,
            iter=4,
            sg=0)

    w2v_model.build_vocab(ingredients_per_recipe, progress_per=1000)
    w2v_model.train(ingredients_per_recipe, total_examples=w2v_model.corpus_count,
                    epochs=20, report_delay=1)
    w2v_model.init_sims(replace=True)


    w1='pumpkin'
    # w1='rum'
    w2v_model.wv.most_similar(positive=w1, topn=10)
      
    
    return w2v_model    
#=========================================================================# 
def get_ingredients_lists(new_ingredients): 
    """
    Get the recipes and retur the ingredients and quantities
    Input:
           List of recipes
    Output:
        List of ingredients per recipes, quantities of ingredients,
        list with all the ingredients
    """       
    count =0
    ingredients_per_recipe = []
    qts_per_recipe = []
    all_ingredients = []
    for drecipe in new_ingredients:
        ings =[]
        qts = []
        for ipos, entry in enumerate(drecipe['ingredients']):
            if drecipe['valid'][ipos]:
                ings.append(entry['text'])
                all_ingredients.append(entry['text'])
                qts.append(drecipe['qty'][ipos]['qty'])
        ingredients_per_recipe.append(ings)
        qts_per_recipe.append(qts)
        count += 1
        if count%500==0:
            logging.info("read {0} ingredients".format(count))
    return ingredients_per_recipe, qts_per_recipe, all_ingredients
#=========================================================================# 
def fit_RandomForest(X_train, y_train):
    """
    Fit a RandomForestClassifier using X_train and y_train
    Input:
        X_train data  and y_train labels
    Outpu:
        randomforest classifier
        
    """
    rforest = RandomForestClassifier(max_depth=10, min_samples_split=10)#, max_features=50)
    rforest.fit(X_train, y_train)
    
    return rforest    
#=========================================================================#
    
#=========================================================================#
    
#=========================================================================#
#load the data
def main(save_models=False):
    logging.info('Start by getting some data! ')


    # Get the cleaned and merged data
    print('Reading full reicpes')
    data_file = '../data/Merged_Recipe.json'
    with open(data_file, 'r') as file:
        new_data = json.load(file)

    data_file = '../data/Merged_Ingredients_only.json'
    with open(data_file, 'r') as file:
        new_data_ings = json.load(file)


    data_file = '../data/Merged_Recipe_Ingredients.json'
    with open(data_file, 'r') as file:
        new_ingredients = json.load(file)
        
    data_file = '../data/Merged_Target.json'
    with open(data_file, 'r') as file:
        target = json.load(file)    

    print('classified {0}/{1}'.format(len(new_data),len(new_data_ings)))

    # set the data into a pandas dataframe to handle the data
    df = pd.DataFrame([new_data_ings,target]).T
    df.columns = ['ingredients','dessert']

    # get the test and the labels
    text = df.ingredients.values
    

    # dimension for the embedding matrix
    embedding_dim = 50
    
    #get the data and a word index
    data, word_index = sequence_data(text)
    # set tha label to categorical                             
    

    # set a word 2 vector model and fit it with all the ingredients
    ingredients_per_recipe, qts_per_recipe, all_ingredients = get_ingredients_lists(new_ingredients)
    
    w2v_model = get_Word2Vec(embedding_dim, ingredients_per_recipe)


    if save_models:
        filename = '../models/w2vec_model.pkl'
        pickle.dump(w2v_model, open(filename, 'wb'))
    
    # use the word 2 vec model to create a data matrix and 
    # fit a random Forest model with it
    data_matrix =[]
    for item in range(len(ingredients_per_recipe)):
        sentence_vec = np.zeros(embedding_dim)
    
        for ingreds in ingredients_per_recipe[item]:
            sentence_vec += w2v_model.wv.__getitem__(ingreds)
    
        data_matrix.append(sentence_vec)
   
    X_train, X_test, y_train, y_test = train_test_split(data_matrix,  
                                                    df.dessert.values.astype(int))

    rforest = fit_RandomForest(X_train, y_train)

    if save_models:
        filename = '../models/forest_w2vec_model.pkl'
        pickle.dump(rforest, open(filename, 'wb'))
        
        filename = '../models/w2vec/X_train.pkl'
        pickle.dump(X_train, open(filename, 'wb'))
        filename = '../models/w2vec/X_test.pkl'
        pickle.dump(X_test, open(filename, 'wb'))
        filename = '../models/w2vec/y_train.pkl'
        pickle.dump(y_train, open(filename, 'wb'))
        filename = '../models/w2vec/y_test.pkl'
        pickle.dump(y_test, open(filename, 'wb'))
    
    y_pred_train = rforest.predict(X_train)
    y_pred_test = rforest.predict(X_test)

    print('train accuracy: {:1.2f}'.format(accuracy_score(y_train, y_pred_train)) )
    print(confusion_matrix(y_train, y_pred_train))
    print('test accuracy: {:1.2f}'.format(accuracy_score(y_test, y_pred_test)) )
    print(confusion_matrix(y_test, y_pred_test))

    pass
#=============================================================================#
    #using word 2 vec to create the matrix, weighted by the amounts
    data_matrix_w =[]
    for item in range(len(ingredients_per_recipe)):
        sentence_vec = np.zeros(embedding_dim)

        for jtem, ingreds in enumerate(ingredients_per_recipe[item]):
            if qts_per_recipe[item][jtem]==0: 
                sentence_vec += w2v_model.wv.__getitem__(ingreds)
            else:
                sentence_vec += w2v_model.wv.__getitem__(ingreds)*qts_per_recipe[item][jtem]
        data_matrix_w.append(sentence_vec)

    data_df = pd.DataFrame(data_matrix_w) 
    data_df['target'] = df.dessert.values.astype(int)
    data_df = data_df.dropna()
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(data_df.drop(['target'],axis=1),  
                                                                data_df['target'])


    rforest_w = fit_RandomForest(X_train_w, y_train_w)
    
    y_pred_train_w = rforest_w.predict(X_train_w)
    y_pred_test_w = rforest_w.predict(X_test_w)
    print('train accuracy: {:1.2f}'.format(accuracy_score(y_train_w, y_pred_train_w)) )
    print(confusion_matrix(y_train_w, y_pred_train_w))
    print('test accuracy: {:1.2f}'.format(accuracy_score(y_test_w, y_pred_test_w)) )
    print(confusion_matrix(y_test_w, y_pred_test_w))

    if save_models:
        filename = '../models/forest_w2vec_weighted_model.pkl'
        pickle.dump(rforest_w, open(filename, 'wb'))
        filename = '../models/w2vec_w/X_train_w.pkl'
        pickle.dump(X_train_w, open(filename, 'wb'))
        filename = '../models/w2vec_w/X_test_w.pkl'
        pickle.dump(X_test_w, open(filename, 'wb'))
        filename = '../models/w2vec_w/y_train_w.pkl'
        pickle.dump(y_train_w, open(filename, 'wb'))
        filename = '../models/w2vec_w/y_test_w.pkl'
        pickle.dump(y_test_w, open(filename, 'wb'))

    label = df.dessert.values
    label = to_categorical(np.asarray(label))
#=============================================================================#
#using the word2vec as embedding
    unique_ingredients = set(all_ingredients)
    embedding_matrix_w2v = get_w2v_embedding_matrix(unique_ingredients, embedding_dim, w2v_model)

    w2v_rnn_model,x_train, y_train, x_val, y_val, X_test, y_test = embedding_LSTM(data, label,embedding_matrix_w2v, 
                                     unique_ingredients, embedding_dim, 0)
    
    if save_models:
        filename = '../models/rnn_w2vec_model.pkl'
        pickle.dump(w2v_rnn_model, open(filename, 'wb'))
        
        filename = '../models/LTSM_w2vec/x_train.pkl'
        pickle.dump(x_train, open(filename, 'wb'))
        filename = '../models/LTSM_w2vec/x_test.pkl'
        pickle.dump(X_test, open(filename, 'wb'))
        filename = '../models/LTSM_w2vec/y_train.pkl'
        pickle.dump(y_train, open(filename, 'wb'))
        filename = '../models/LTSM_w2vec/y_test.pkl'
        pickle.dump(y_test, open(filename, 'wb'))
        
#=============================================================================#    
# use the gloVe embeddings
    embedding_matrix = get_glove_embedding_matrix(word_index, embedding_dim)

    glove_rnn_model,x_train, y_train, x_val, y_val, X_test, y_test  = embedding_LSTM(data, label,embedding_matrix, 
                                     word_index, embedding_dim,1)

    if save_models:
        filename = '../models/rnn_glove_model.pkl'
        pickle.dump(glove_rnn_model, open(filename, 'wb'))
        
        filename = '../models/LTSM_glove/x_train.pkl'
        pickle.dump(x_train, open(filename, 'wb'))
        filename = '../models/LTSM_glove/x_test.pkl'
        pickle.dump(X_test, open(filename, 'wb'))
        filename = '../models/LTSM_glove/y_train.pkl'
        pickle.dump(y_train, open(filename, 'wb'))
        filename = '../models/LTSM_glove/y_test.pkl'
        pickle.dump(y_test, open(filename, 'wb'))
#=============================================================================#    
# finally test with a random embedding matrix
    embedding_rand_matrix = get_random_embedding_matrix(word_index, embedding_dim)
    random_rnn_model,x_train, y_train, x_val, y_val, X_test, y_test  = embedding_LSTM(data, label,embedding_rand_matrix, 
                                     word_index, embedding_dim,1,trainable=True)
    
    if save_models:
        filename = '../models/rnn_trainable_model.pkl'
        pickle.dump(random_rnn_model, open(filename, 'wb'))
        
        filename = '../models/LTSM_rand/x_train.pkl'
        pickle.dump(x_train, open(filename, 'wb'))
        filename = '../models/LTSM_rand/x_test.pkl'
        pickle.dump(X_test, open(filename, 'wb'))
        filename = '../models/LTSM_rand/y_train.pkl'
        pickle.dump(y_train, open(filename, 'wb'))
        filename = '../models/LTSM_rand/y_test.pkl'
        pickle.dump(y_test, open(filename, 'wb'))

    print('DONE!!!')        

    


if __name__ == "__main__":
    
    save_models = True
    main(save_models)



## Word Frequencies with TfidVectorizer

## Word Hashing 