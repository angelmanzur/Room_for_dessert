#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 21:28:21 2019

@author: angelmanzur
"""

######
#Create some confusion matrices

#####
#BAse model w2vec with a random forest
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import average_precision_score

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=None):
    #=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots(figsize=(10,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap,
                   vmin =0, vmax=1.)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Fraction of dessets', fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes)
    #cbar = fig.colorbar(im, ax=ax2)
    #åcbar.set_label('booy', fontsize=12)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.title(title, fontsize=26)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('Predicted label', fontsize=22)
    plt.ylabel('True label', fontsize=22)
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),fontsize=12,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def get_data(model_test,model_name, model_file, xtrain_file, xtest_file, ytrain_file, ytest_file):
    
    model = pickle.load(open(model_file, 'rb'))
    x_train = pickle.load(open(xtrain_file, 'rb'))
    x_test = pickle.load(open(xtest_file, 'rb'))
    y_train = pickle.load(open(ytrain_file,'rb'))
    y_test = pickle.load(open(ytest_file,'rb'))
    
    
    
    
    y_pred_train = model.predict(x_train)
    y_pred_test  = model.predict(x_test)
    
    flat_y_test = []
    flat_y_train = []
    flat_y_pred_train = []
    flat_y_pred_test = []
    if model_test>1 and model_test<5:
   # if y_test.shape[1]>0:
        for yvec in y_test:
            pos = np.where(yvec == yvec.max()) 
            flat_y_test.append(pos[0][0])
        for yvec in y_train:
            pos = np.where(yvec == yvec.max()) 
            flat_y_train.append(pos[0][0])
            
        for yvec in y_pred_train:
            pos = np.where(yvec == yvec.max()) 
            flat_y_pred_train.append(pos[0][0])     

        for yvec in y_pred_test:
            pos = np.where(yvec == yvec.max()) 
            flat_y_pred_test.append(pos[0][0]) 
    else: 
        flat_y_test = y_test
        flat_y_train = y_train
        flat_y_pred_test = y_pred_test
        flat_y_pred_train = y_pred_train
    
    
    
    if model_name == 'random':
        for item, val in enumerate(flat_y_test):
            flat_y_test[item] = np.random.randint(4)
        for item, val in enumerate(flat_y_train):
            flat_y_train[item] = np.random.randint(4)    
            
            
    categories = ['cake', 'cookies', 'pie', 'pudding']


    plot_confusion_matrix(flat_y_train, flat_y_pred_train,
                          categories,
                          cmap=plt.cm.Blues,normalize=True)
    plt.savefig('../figs/{0}_conf_matrix_train.png'.format(model_name))
    
    plot_confusion_matrix(flat_y_test, flat_y_pred_test,
                          categories,
                          cmap=plt.cm.Blues,normalize=True)
    plt.savefig('../figs/{0}_conf_matrix_test.png'.format(model_name))
    
    acc_train_score = accuracy_score(flat_y_train, flat_y_pred_train)
    acc_test_score = accuracy_score(flat_y_test, flat_y_pred_test)
#    ap_score_train = average_precision_score(y_train, y_pred_train)
#    ap_score_test  = average_precision_score(y_test, y_pred_test)
    accuracy_results = 'Accuracy score: \t train {:1.2f}\t test {:1.2f}'.format(acc_train_score, 
          acc_test_score)
    print(accuracy_results)
#    print('Avg Prec score: \t train {:1.2f}\t test {:1.2f}'.format(ap_score_train,ap_score_test))
 
    result_file = '../results/Fit_{0}.txt'.format(model_name)
    print(classification_report(flat_y_train, flat_y_pred_train))
    print(classification_report(flat_y_test, flat_y_pred_test))
    
    file1 = open(result_file, 'w+')
    file1.write(accuracy_results)
    file1.write('\n========================================\n')
    file1.write('Classification Report train dataset\n')
    file1.write(classification_report(flat_y_train, flat_y_pred_train))
    file1.write('\n========================================\n')
    file1.write('Classification Report test dataset\n')
    file1.write(classification_report(flat_y_test, flat_y_pred_test))

    file1.close()


model_test = 3

if model_test == 0:
# w2vec on a random forest base model
    model_file = '../models/forest_w2vec_model.pkl'
    x_train_file = '../models/w2vec/X_train.pkl'
    x_test_file = '../models/w2vec/X_test.pkl'
    y_train_file = '../models/w2vec/y_train.pkl'
    y_test_file = '../models/w2vec/y_test.pkl'
    model_name = 'w2v_rforest'
elif model_test == 1:
# weighted w2vec on a random forest
    model_file = '../models/forest_w2vec_weighted_model.pkl'
    x_train_file = '../models/w2vec_w/X_train_w.pkl'
    x_test_file = '../models/w2vec_w/X_test_w.pkl'
    y_train_file = '../models/w2vec_w/y_train_w.pkl'
    y_test_file = '../models/w2vec_w/y_test_w.pkl'
    model_name = 'w2v_w_rforest'
elif model_test == 2:
    # rnn with w2vec  embeding
    model_file = '../models/rnn_w2vec_model.pkl'
    x_train_file = '../models/LTSM_w2vec/x_train.pkl'
    x_test_file = '../models/LTSM_w2vec/x_test.pkl'
    y_train_file = '../models/LTSM_w2vec/y_train.pkl'
    y_test_file = '../models/LTSM_w2vec/y_test.pkl'
    model_name = 'w2v_rnn'
elif model_test == 3:
    # rnn with gloVe embedding
    model_file = '../models/rnn_glove_model.pkl'
    x_train_file = '../models/LTSM_glove/x_train.pkl'
    x_test_file = '../models/LTSM_glove/x_test.pkl'
    y_train_file = '../models/LTSM_glove/y_train.pkl'
    y_test_file = '../models/LTSM_glove/y_test.pkl'
    model_name = 'glove_rnn'
elif model_test == 4:
    # rnn with random embedding
    model_file= '../models/rnn_trainable_model.pkl'
    x_train_file= '../models/LTSM_rand/x_train.pkl'
    x_test_file = '../models/LTSM_rand/x_test.pkl'
    y_train_file = '../models/LTSM_rand/y_train.pkl'
    y_test_file = '../models/LTSM_rand/y_test.pkl'
    model_name = 'rand_rnn'
elif model_test == 5:
    # rnn with random embedding
    model_file = '../models/forest_w2vec_model.pkl'
    x_train_file = '../models/w2vec/X_train.pkl'
    x_test_file = '../models/w2vec/X_test.pkl'
    y_train_file = '../models/w2vec/y_train.pkl'
    y_test_file = '../models/w2vec/y_test.pkl'
    model_name = 'random'     
        
get_data(model_test, model_name, model_file, x_train_file, x_test_file, y_train_file, y_test_file)




