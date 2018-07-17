# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:46:29 2018

@author: shivanshu_dikshit
"""

#importing the necessary libraries to be used
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#importing the dataset by the help of pnadas library
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(train.info())
print("_________------------------__________________")
print(test.info())

"""
as we can clearly see from the info that there are missing values in the train data in the columns "age"
, "cabin" and "embarked"
similarly in the test data there are missing values in the columns "age", "fare" and "cabin"
"""
print(train.head(10))
print("------------------------------____________________________________-------------------------------")
print(test.head(10))


"""
as we need to process both the train and test set so we would merge them and process them together
first of all we will take care of the missing values and the we will convert the data into their 
required representation
"""
combined = [train, test]
for data in combined:
    
    #filling the missing values of tha age column with the median of age
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Age'] = data['Age'].astype(int)
    
    #filling the missing value of fair in the test data
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    
    #filling the value in embarked with the most occuring one that is "S"
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Embarked'] = data['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
    
    #chaning the values in sex column into numeric values
    data['Sex'] = data['Sex'].map({'female':0, 'male':1}).astype(int)
    
    #adding a new feature to the data by combining "parch" and "sibsp" and adding 1 as the person itself
    data['family'] = data['Parch'] + data['SibSp'] + 1

#rechecking the data
print(train.head(10))
print("-------------------____________________--------------------")
print(test.head(10))
print("-------------------------_______________________---------------------------")
print(train.info())
print("-------------------------_______________________---------------------------")
print(test.info())

"""
creating the target variable and the train features from the train data
creating the test features from the test data
"""
target = train['Survived']
train_features = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'family']]
test_features = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'family']]

#visualising the created variables
print(train_features.head(10))
print("________________________------------------------_____________________________________")
print(train_features.head(10))
print("________________________------------------------_____________________________________")
print(train_features.info())

#fitting the model to our data
forest = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=1)
model = forest.fit(train_features, target)

#making the prediction through help of our created model and testing it on test data
prediction = model.predict(test_features)

submission = pd.DataFrame({'PassengerId':data['PassengerId'], "Survived":prediction})
submission.to_csv('submission.csv', index = False)

submission = pd.read_csv('submission.csv')
print(submission.head(10))