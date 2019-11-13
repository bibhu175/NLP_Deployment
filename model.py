# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:58:17 2019

@author: BIBHUTI
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle

df = pd.read_csv("spam.csv", encoding = "latin-1")
df.head()

df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1, inplace = True)

# feature & labels
df = df.rename(columns = {"v1":"label", "v2":"message"})
df['status'] = df['label'].map({'ham': 0, 'spam' :1})
X = df['message']
y = df['label']
 
# Extract feature with CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

pickle.dump(cv, open('tranform.pkl', 'wb'))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
filename = 'NLP_model.pkl'

 