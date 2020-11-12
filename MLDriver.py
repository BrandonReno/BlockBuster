import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
	print("Random Forest Within One Accuracy: ",withinOne(np.argmax(RFClassifier.predict(TestingInput), axis=1), TestingOutput.argmax(axis=1)))
	cm = confusion_matrix(TestingOutput.argmax(axis=1), Prediction.argmax(axis=1)) 
	return cm

def TestKNN():
	KnnClassifier = KNeighborsClassifier(n_neighbors = 3).fit(TrainingInput, TrainingOutput)  
	Predictions = KnnClassifier.predict(TestingInput)  
	print("KNN Within One Accuracy: ", withinOne(np.argmax(KnnClassifier.predict(TestingInput), axis=1), TestingOutput.argmax(axis=1)))
	acc = KnnClassifier.score(TestingInput, TestingOutput)
	cm = confusion_matrix(TestingOutput.argmax(axis=1), Predictions.argmax(axis=1)) 
	return cm
	

def TestSVM():
	SVMClassifier = SVC(kernel = 'linear', C = 1).fit(TrainingInput, TrainingOutput) 
	Predicitons = SVMClassifier.predict(TestingInput) 
	accuracy = SVMClassifier.score(TestingInput, TestingOutput) 

	print(accuracy)

def TestNNModel():
	model = CreateNNmodel()
	model.fit(TrainingInput, TrainingOutput, epochs=1000, verbose=0,callbacks=callback, validation_split=.25)
	acc = model.evaluate(TestingInput,TestingOutput)
	results = np.argmax(model.predict(TestingInput), axis=1)
	print("NN Within One Accuracy:" ,withinOne(np.argmax(model.predict(TestingInput), axis=1), TestingOutput.argmax(axis=1)))
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



def ConfusionMatrix(cm):
	categories = [ 'Above Average','Below Average','Blockbuster', 'Bust',]
	df_cm = pd.DataFrame(cm, index = categories, columns = categories)
	plt.figure(figsize = (10,7))
	sns.heatmap(df_cm, annot=True, cmap = 'Blues')
	plt.show()


def TestModels():
	ConfusionMatrix(TestKNN())
	ConfusionMatrix(TestRandomForest())
	ConfusionMatrix(TestNNModel())

TestModels()







