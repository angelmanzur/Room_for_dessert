#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:17:40 2019

@author: angelmanzur
"""

import pickle
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

    
pickle_file = '../models/tokenizer.pkl'
tokenizer = pickle.load(open(pickle_file, 'rb'))
    

pickle_file = '../models/sequences.pkl'
sequences = pickle.load(open(pickle_file, 'rb')) 
    
pickle_file = '../models/word_index.pkl'
word_index = pickle.load(open(pickle_file, 'rb'))
#load the w2vec pickle model

filename = '../models/w2vec_model.pkl'
w2v_model = pickle.load(open(filename, 'rb'))



text_4 = ['granulated','sugar','corn','oil','Beaters', 'egg', 
          'substitute','orange','rind','orange','juice','baking','powder','rum']
seq = [25, 1, 71, 23, 12, 504, 12, 146, 39, 155, 39, 27, 5, 10, 3]

myseq = []
for word in text_4:
    try: 
        myseq.append(word_index[word.lower()])
    except:
        myseq.append(0)

embedding_dim = 100
x_data = np.zeros(embedding_dim)
myseq_len =len(myseq)+1
for x in range(-1, -myseq_len,-1):
    x_data[x] = int(myseq[x])
    
hold_data = np.empty([1,100])
hold_data[0] = x_data
print(hold_data)
model_file = '../models/rnn_w2vec_model.pkl'
model = pickle.load(open(model_file, 'rb'))
y_pred = model.predict(hold_data)

dessert_cat = np.where(y_pred[0] == y_pred[0].max()) 

categories = ['cake', 'cookies', 'pie', 'pudding']

print('looks like you are makingg: {}'.format(categories[dessert_cat[0][0]]))

