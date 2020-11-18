import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns
import os

#This makes Tensorflow run on my CPU instead of GPU: Gpu is very laggy because of no batch size
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#These will hold accuracies for plotting after the tests
NNAccuracies = []
NNWOAccuracies = []

KnnAccuracies = []
KnnWOAccuracies = []

RFAccuracies = []
RFWOAccuracies = []



# load dataset
dataframe = pd.read_csv("NNInputTest3.csv", header=0)
dataset = dataframe.values

#Preprocess the data with normalization since distribution is not gaussian and ranges are far different
scaler = MinMaxScaler() 
dataset = scaler.fit_transform(dataset)

#Cutoff is used to specify the amount of data that will be used for training
Cutoff = int(len(dataset)*.9)
Features = 3


#Training Features and Labels
TrainingInput = dataset[:Cutoff,0:Features]
TrainingOutput = dataset[:Cutoff,Features:]

#Testing Features and Labels
TestingInput = dataset[Cutoff:,0:Features]
TestingOutput = dataset[Cutoff:,Features:]

#For NN
callback = EarlyStopping(monitor='accuracy', patience=30)

def CreateNNmodel():
	# create model
	model = Sequential()
	model.add(Dense(10, input_dim=Features, activation='relu'))
	model.add(Dense(12, activation= 'relu'))
	model.add(Dense(14, activation= 'relu'))
	model.add(Dense(4, activation = 'softmax'))
	# Compile model
	opt = tf.keras.optimizers.Adam(learning_rate=0.001)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

def TestRandomForest():
	RFClassifier = RandomForestClassifier(n_estimators=200)
	RFClassifier.fit(TrainingInput, TrainingOutput)
	Prediction = RFClassifier.predict(TestingInput)
	acc = accuracy_score(TestingOutput, Prediction)
	WOacc = withinOne(np.argmax(RFClassifier.predict(TestingInput), axis=1), TestingOutput.argmax(axis=1))
	print("RF Accuracy: ", acc, "Within One: ", WOacc)
	RFAccuracies.append(acc)
	RFWOAccuracies.append(WOacc)
	cm = confusion_matrix(TestingOutput.argmax(axis=1), Prediction.argmax(axis=1)) 
	return cm

def TestKNN():
	KnnClassifier = KNeighborsClassifier(n_neighbors = 3).fit(TrainingInput, TrainingOutput)  
	Predictions = KnnClassifier.predict(TestingInput)  
	WOacc = withinOne(np.argmax(KnnClassifier.predict(TestingInput), axis=1), TestingOutput.argmax(axis=1))
	acc = KnnClassifier.score(TestingInput, TestingOutput)
	print("KNN Accuracy: ", acc, "Within One: ", WOacc)
	KnnAccuracies.append(acc)
	KnnWOAccuracies.append(WOacc)
	cm = confusion_matrix(TestingOutput.argmax(axis=1), Predictions.argmax(axis=1)) 
	return cm
	
def TestNNModel():
	model = CreateNNmodel()
	#cv = KFold(n_splits=10, random_state=1, shuffle=True)
	model.fit(TrainingInput, TrainingOutput, epochs=600, verbose=0)
	""" scores = cross_val_score(model, TrainingInput, TrainingOutput, scoring='accuracy', cv=cv, n_jobs=-1)
	print('Accuracy: %.3f (%.3f)' % (mean(scores))) """
	acc = model.evaluate(TestingInput,TestingOutput)
	WOacc = withinOne(np.argmax(model.predict(TestingInput), axis=1), TestingOutput.argmax(axis=1))
	print("NN Accuracy: ", acc[1], "Within One: ", WOacc)
	NNAccuracies.append(acc[1])
	NNWOAccuracies.append(WOacc)
	results = np.argmax(model.predict(TestingInput), axis=1)
	cm = confusion_matrix(TestingOutput.argmax(axis=1), results) 
	return cm

def withinOne(prediction, actual):
	acc = 0
	total = 0
	for p, a in zip(prediction,actual):
		if (p + 1 == a or p - 1 == a or p == a):
			acc +=1
		total +=1
	return acc/total

def ConfusionMatrix(cm, title):
	categories = [ 'Above Average','Below Average','Blockbuster', 'Bust',]
	df_cm = pd.DataFrame(cm, index = categories, columns = categories)
	plt.figure(figsize = (10,7))
	plt.title(title)
	sns.heatmap(df_cm, annot=True, cmap = 'Blues')
	plt.show()


def TestModels():
		ConfusionMatrix(TestKNN(), "Knn Confusion Matrix")
		ConfusionMatrix(TestRandomForest(), "Random Forest Confusion Matrix")
		ConfusionMatrix(TestNNModel(), "Neural Network Confusion Matrix")

TestModels()







