# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:00:28 2021

@author: Jonathan Frassineti and Pierpaolo Vecchi
"""

# =============================================================================
# CLASSIFICATION PROBLEM FOR THE SONAR DATASET
# =============================================================================

###################
###################
###################

# =============================================================================
# IMPORT SOME LIBRARIES AND MODULES
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import ShuffleSplit 

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier                             
from sklearn.neighbors import KNeighborsClassifier                          
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis        
from sklearn.naive_bayes import GaussianNB                                  
from sklearn.svm import SVC    
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# LOAD THE DATASET "SONAR.CSV"
# =============================================================================

dataframe = pd.read_csv("sonar.csv", header = None)
data = dataframe.values

# =============================================================================
# SPLIT THE DATA INTO INPUTS (X_input) AND OUTPUTS (Y_output)
# OUPUTS ARE "M" FOR MINES and "R" FOR ROCKS
# =============================================================================

X_input = data[:,0:60]
Y_output = data[:,60]

# =============================================================================
# PLOT THE LOADED DATA
# =============================================================================

dataframe.plot.box(figsize=(14,8), xticks=[])
plt.title('Boxplots of all frequency bins', fontsize = 30)
plt.xlabel('Frequency bin',fontsize = 24)
plt.ylabel('Power spectral density (normalized)',fontsize = 24)
plt.tick_params(labelsize=24)
plt.show()

# =============================================================================
# TRANSFORM OUPUT VALUES IN "0" FOR "M" ANS "1" FOR "R"
# =============================================================================

encoder = preprocessing.LabelEncoder()
encoder.fit(Y_output)
encoded_Y_ouput = encoder.transform(Y_output)

for key, value in enumerate(encoder.classes_):
    print(value, "=",key)
    
###################
###################
###################
    
# =============================================================================
# NOW TRY 3 DIFFERENT EVALUATION METHODS TO TEST THE ML ALGORITHM
# 1) SPLIT INTO TRAIN AND TEST DATA
# 2) K-FOLD CROSS-VALIDATION
# 3) REPEATED RANDOM TEST-TRAIN SPLITS
# =============================================================================

# =============================================================================
# 1) SPLIT INTO TRAIN AND TEST DATA
# =============================================================================
    
size = 0.39
seed = 8

X_train, X_test, Y_train, Y_test = train_test_split(X_input, Y_output, test_size=size, random_state=seed)
model = LogisticRegression()            
model.fit(X_train, Y_train)             
results = model.score(X_test, Y_test)    
print("Accuracy: %.3f%%" % (results*100.0))

# =============================================================================
# 2) K-FOLD CROSS-VALIDATION
# =============================================================================
  
folds = 100
seed = 8

kfold = KFold(n_splits=folds, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, X_input, Y_output, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

# =============================================================================
# 3) REPEATED RANDOM TEST-TRAIN SPLITS
# =============================================================================
  
splits = 100
size = 0.39
seed = 8

kfold = ShuffleSplit(n_splits=splits, test_size=size, random_state=seed)
model = LogisticRegression()
results = cross_val_score(model, X_input, Y_output, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

# =============================================================================
# FROM NOW ON, WE CHOOSE THE K-FOLD CROSS-VALIDATION
# =============================================================================

###################
###################
###################

# =============================================================================
# WE NOW COMPARE DIFFERENT ML ALGORITHMS
# FIRST, PREPARE THE MODELS
# =============================================================================

models = []
models.append(( 'LR'   , LogisticRegression(solver='lbfgs', max_iter=500)))
models.append(( 'LDA'  , LinearDiscriminantAnalysis()))
models.append(( 'KNN'  , KNeighborsClassifier()))
models.append(( 'CART' , DecisionTreeClassifier()))
models.append(( 'NB'   , GaussianNB()))
models.append(( 'SVM'  , SVC()))
models.append(( 'ADA'  , AdaBoostClassifier()))
models.append(( 'RFC'  , RandomForestClassifier()))
models.append(( 'GBC'  , GradientBoostingClassifier()))                                            

# =============================================================================
# EVALUATE EACH MODEL ONE BY ONE
# =============================================================================

seed = 8
splits = 15
results = []
names = []
scoring = 'accuracy'
for name, model in models:
  kfold = KFold(n_splits=splits, random_state=seed)
  cv_results = cross_val_score(model, X_input, Y_output, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)

# =============================================================================
# PLOT THE ALGORTIHM COMPARISON IN A BOXPLOT, WITH A TABLE BELOW
# =============================================================================

columns = names
rows = ['Accuracy (%)','Variance (%)']

cell_text = np.zeros((2,9))

for i in range(9):
    cell_text[0][i] = round(results[i].mean(),3)
    cell_text[1][i] = round(results[i].std(),3)
    

fig, ax = plt.subplots(1,1, figsize = (14,8))
ax.boxplot(results)
ax.set_ylabel('Accuracy (%)',fontsize = 24)
ax.xaxis.set_visible(False)
ax.tick_params(labelsize=24)
ax.set_title('Algorithm comparison', fontsize = 30)
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns,
                      loc= 'bottom')
the_table.set_fontsize(20)
the_table.scale(1, 2) 
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.show()