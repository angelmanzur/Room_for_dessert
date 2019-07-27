#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:10:32 2019

@author: angelmanzur
"""



import pickle
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

filename = '../models/w2vec_model.pkl'
w2v_model = pickle.load(open(filename, 'rb'))

vector = w2v_model.wv.__getitem__('butter')
vec_size = len(vector)

word = 'pumpkin'

w2v_model.wv.most_similar(positive=word, topn=10)


#data_1000 = standardized_data[0:1000,:]
#labels_1000 = labels[0:1000]
#model = TSNE(n_components=2, random_state=0)
#tsne_data = model.fit_transform(data_1000)
#tsne_data = np.vstack((tsne_data.T, labels_1000)).T
#tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
#sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
#plt.show()


arrays = np.empty((0,vec_size), dtype='f')
    # set the original word,     
word_labels = [word]
color_list = ['blue']
    # append the vector for the test word
arrays = np.append(arrays, w2v_model.wv.__getitem__([word]),axis=0)
    
    ####### get the closest words and make then blue
    #get a list of the most similar words
close_words = w2v_model.wv.most_similar([word],topn=10)
    
for word_score in close_words:
        # for each word get the vector  
    word_vector = w2v_model.wv.__getitem__([word_score[0]])
        # get the name
    word_labels.append(word_score[0])
    color_list.append('red')
    arrays = np.append(arrays, word_vector, axis=0)


tsne_model = TSNE(n_components=2, random_state=0,n_iter=500)
tsne_data = tsne_model.fit_transform(arrays)

tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2"))
tsne_df['Word'] = word_labels
tsne_df['Colors'] = color_list

fig, _ = plt.subplots()
fig.set_size_inches(8,6)
    
p1 = sns.regplot(data=tsne_df, 
                    x='Dim_1',y='Dim_2',fit_reg=False,
                    marker='o',
                    scatter_kws={'s':50,
                                'facecolors':tsne_df['Colors']})
    
for line in range(tsne_df.shape[0]):
    p1.text(tsne_df['Dim_1'][line],tsne_df['Dim_2'][line],#df['z'][line],
           '  '+tsne_df['Word'][line].title(),
           horizontalalignment='left',
            verticalalignment='bottom',size='large',
            color=tsne_df['Colors'][line],weight='normal'
           ).set_size(10)
#     fig = plt.figure(figsize=(8,6))
#     ax = Axes3D(fig)  

#     ax.scatter(df['x'], df['y'], df['z'], marker='o')
plt.xlim(tsne_df['Dim_1'].min()-40,tsne_df['Dim_1'].max()+40 )
plt.ylim(tsne_df['Dim_2'].min()-40,tsne_df['Dim_2'].max()+40 )
#     plt.zlim(Y[:,2].min(), Y[:,2].max() )
plt.title('Food pairing for {}'.format(word.title()))


#sns.FacetGrid(tsne_df, size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()
             
    #Y = PCA(n_components=2).fit_transform(arrays)
