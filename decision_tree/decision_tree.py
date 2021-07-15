from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import tree
from matplotlib.pyplot import figure
from dtreeviz.trees import dtreeviz


def prep_data(X_pd, y_pd):
    '''Take two dataframes, and convert to numpy and do train test split'''
    
    X = X_pd.to_numpy()
    y = y_pd.to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=.25, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test 
  
  
def fit_decision_tree(X_train, y_train):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    
    print("Depth: ", dt.get_depth())
    print("Number of leaves: ", dt.get_n_leaves())
    
    return dt
  
  
def visualize_tree(dt, data, type=0):
    
    if type == 0:
        fig = figure(figsize=(25,20))
        _ = tree.plot_tree(dt, feature_names=data.feature_names, 
                           class_names=data.target_names,
                          filled=True)
        
    if type == 1:
        viz = dtreeviz(dt, data.data, data.target,
                       target_name='target',
                       feature_names=data.feature_names, 
                       class_names=data.target_names.tolist()
                      )
        viz.view()
        
        
 def evaluate(dt, X_test, y_test):
    print("Score: ", dt.score(X_test, y_test))
    plot_confusion_matrix(dt, X_test, y_test)
    
    
