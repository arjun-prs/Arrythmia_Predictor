# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('DS1_signals.csv')
X = dataset.iloc[:, 1:].values
# #Converting words to integer values
# def convert_to_int(word):
#     word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
#                 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
#     return word_dict[word]

#X['experience'] = X['experience'].apply(lambda x : float(x))

y = dataset.iloc[:,0]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

# from sklearn.linear_model import LinearRegression
# # regressor = LinearRegression()
from sklearn.svm import SVC
classifier = SVC()
#Fitting model with trainig data
classifier.fit(X, y)

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))