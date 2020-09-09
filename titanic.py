import pandas as pd
import random as rnd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#Read in data
trainingData = pd.read_csv('train.csv')
testingData = pd.read_csv('test.csv')
#Combine data for processing that needs to happen on both sets
combinedData = [trainingData, testingData]

#Preprocess Data

##Step 1 - From HW1 we know that features Cabin and Ticket aren't useful, so we will drop them immediately
#We will also drop PassengerID from both as well since it is not an important feature
trainingData = trainingData.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
testingData = testingData.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
#Update the combined data array
combinedData = [trainingData, testingData]

##Step 2 - Fill missing Data
#The Embarked feature in the trainingData set is missing two values, so fill with the most common
mostCommonEmbarkationPort = trainingData.Embarked.dropna().mode()[0]
trainingData['Embarked'] = trainingData['Embarked'].fillna(mostCommonEmbarkationPort)

#The 'Fare' feature in the testingData set is missing one value, so fill with the median of the feature
medianFare = testingData['Fare'].dropna().median()
testingData['Fare'] = testingData['Fare'].fillna(medianFare)

##Step 3 - Loop through the data and preprocess
for data in combinedData:
	#Create new feature called NamePrefix and extract the title from each name (ie Mr., Mrs., ...)
	data['NamePrefix'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
	#Combine prefix's with same meaning but different spelling (mlle & miss, ms & miss, mme & mrs)
	data['NamePrefix'] = data['NamePrefix'].replace('Mlle', 'Miss')
	data['NamePrefix'] = data['NamePrefix'].replace('Ms', 'Miss')
	data['NamePrefix'] = data['NamePrefix'].replace('Mme', 'Mrs')
	#Replace the names of all entries with less than 7 people with the value 'Other'
	data['NamePrefix'] = data['NamePrefix'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Other')
	#Convert 'NamePrefix' categorical feature to ordinal
	data['NamePrefix'] = data['NamePrefix'].map( {'Other' : 0, 'Master' : 1, 'Miss' : 2, 'Mr' : 3, 'Mrs' : 4} )

	#Convert 'Sex' categorical feature to ordinal
	data['Sex'] = data['Sex'].map( {'male' : 0, 'female' : 1} ).astype(int)
	#Rename to 'Gender'
	data.rename(columns={'Sex': 'Gender'}, inplace=True)

	#Need to fill in missing Age values
	#From HW1 we found a correlation between Age, Gender, and Pclass by plotting six graphs
	#We are going to fill in missing values by using the median value of the Age based on the position in the Pclass/Gender chart we created
	for i in range(0, 2):
		for j in range(0, 3):
			ageFromCorrelatedChart = data[(data['Gender'] == i) & (data['Pclass'] == j+1)]['Age'].dropna()
			data.loc[(data.Age.isnull()) & (data.Gender == i) & (data.Pclass == j+1), 'Age'] = int( (ageFromCorrelatedChart.median() * 2.0) + 0.5) / 2.0
	data['Age'] = data['Age'].astype(int)

	#Convert 'Age' values into ordinal data based on Age Groups found in HW1
	data.loc[ (data['Age'] <= 16), 'Age'] = 0
	data.loc[ (data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
	data.loc[ (data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
	data.loc[ (data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
	data.loc[ (data['Age'] > 64), 'Age'] = 4

	#Create new feature 'Alone' to indicate if a person is travelling alone or not.
	data['Alone'] = 0
	#Set the feature to 1 (Meaning traveller is alone) if 'SibSp' + 'Parch' == 0
	data.loc[(data['SibSp'] + data['Parch']) == 0, 'Alone'] = 1

	#Convert 'Embarked' categorical feature to ordinal
	data['Embarked'] = data['Embarked'].map( {'C' : 0, 'Q' : 1, 'S' : 2} ).astype(int)

	#Convert 'Fare' values into ordinal data
	#Fare Groups found by doing a qcut on the data and plotting it against the 'Survival' feature
	#The four groups was selected because it gave a good data split and a group with 58% survival rate
	data.loc[ (data['Fare'] <= 7.91), 'Fare'] = 0
	data.loc[ (data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
	data.loc[ (data['Fare'] > 14.454) & (data['Fare'] <= 31.0), 'Fare'] = 2
	data.loc[ (data['Fare'] > 31.0), 'Fare'] = 3
	data['Fare'] = data['Fare'].astype(int)

##Step 4 - Drop unimportant features
#Dropping Name, SibSp, Parch features
trainingData = trainingData.drop(['Name', 'SibSp', 'Parch'], axis=1)
testingData = testingData.drop(['Name', 'SibSp', 'Parch'], axis=1)
#Update the combined data array
combinedData = [trainingData, testingData]

##Step 5 - Train and classify data
print("Preprocessed Training Data")
print(trainingData.head(5))

print("Preprocessed Testing Data")
print(testingData.head(5))

#Split data for classification
X_train = trainingData.drop(['Survived'], axis=1)
y_train = trainingData['Survived']
X_test = testingData.copy()

#Decision Tree Classifier
decisionTreeClassifier = DecisionTreeClassifier()
decisionTreeClassifier.fit(X_train, y_train)
y_predictions = decisionTreeClassifier.predict(X_test)
score = cross_val_score(decisionTreeClassifier, X_train, y_train, cv=5, scoring='accuracy')

fig, axes = plt.subplots(nrows = 1,ncols = 1, figsize = (4,4), dpi=300)
tree.plot_tree(decisionTreeClassifier)
fig.savefig('decisionTree.png')

print("Decision Tree Classifier Accuracy")
print(score.mean())

#Random Forest Classifier
randomForestClassifier = RandomForestClassifier()
randomForestClassifier.fit(X_train, y_train)
y_predictions = randomForestClassifier.predict(X_test)
randomForestClassifier.score(X_train, y_train)
score = cross_val_score(randomForestClassifier, X_train, y_train, cv=5, scoring='accuracy')

print("Random Forest Classifier Accuracy")
print(score.mean())
