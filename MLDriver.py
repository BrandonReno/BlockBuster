import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns
import os

#These will hold accuracies for plotting after the tests
NNAccuracies = []
NNWOAccuracies = []

KnnAccuracies = []
KnnWOAccuracies = []

RFAccuracies = []
RFWOAccuracies = []

#This makes Tensorflow run on my CPU instead of GPU: Gpu is very laggy because of no batch size
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load dataset
dataframe = pd.read_csv("NNInput.csv", header=0)
dataset = dataframe.values

Cutoff = 3300

#Training Features and Labels
TrainingInput = dataset[:Cutoff,0:8]
TrainingOutput = dataset[:Cutoff,8:]

#Testing Features and Labels
TestingInput = dataset[Cutoff:,0:8]
TestingOutput = dataset[Cutoff:,8:]

#For NN
callback = EarlyStopping(monitor='accuracy', patience=30)

def CreateNNmodel():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=8, activation='relu'))
	model.add(Dense(4, activation = 'softmax'))
	# Compile model
	opt = tf.keras.optimizers.Adam(learning_rate=0.001)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

def TestRandomForest():
	RFClassifier = RandomForestClassifier(n_estimators=500)
	RFClassifier.fit(TrainingInput, TrainingOutput)
	Prediction = RFClassifier.predict(TestingInput)
	acc = accuracy_score(TestingOutput, Prediction)
	WOacc = withinOne(np.argmax(RFClassifier.predict(TestingInput), axis=1), TestingOutput.argmax(axis=1))
	RFAccuracies.append(acc)
	RFWOAccuracies.append(WOacc)
	cm = confusion_matrix(TestingOutput.argmax(axis=1), Prediction.argmax(axis=1)) 
	return cm

def TestKNN():
	KnnClassifier = KNeighborsClassifier(n_neighbors = 3).fit(TrainingInput, TrainingOutput)  
	Predictions = KnnClassifier.predict(TestingInput)  
	WOacc = withinOne(np.argmax(KnnClassifier.predict(TestingInput), axis=1), TestingOutput.argmax(axis=1))
	acc = KnnClassifier.score(TestingInput, TestingOutput)
	KnnAccuracies.append(acc)
	KnnWOAccuracies.append(WOacc)
	cm = confusion_matrix(TestingOutput.argmax(axis=1), Predictions.argmax(axis=1)) 
	return cm
	
def TestNNModel():
	model = CreateNNmodel()
	model.fit(TrainingInput, TrainingOutput, epochs=500, verbose=0, validation_split=.25)
	acc = model.evaluate(TestingInput,TestingOutput)
	print(acc)
	WOacc = withinOne(np.argmax(model.predict(TestingInput), axis=1), TestingOutput.argmax(axis=1))
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
		else:
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

		""" print(mean(NNWOAccuracies))
		print(mean(NNAccuracies))
		print(mean(RFWOAccuracies))
		print(mean(RFAccuracies))
		print(mean(KnnAccuracies))
		print(mean(KnnWOAccuracies)) """

TestModels()







