import pandas as pd
import numpy as np
import tensorflow as tf
from kerastuner.tuners import Hyperband
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from statistics import mean
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

LOG_DIR=f"{int(time.time())}"

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
dataframe = pd.read_csv("NNInputFull.csv", header=0)
dataset = dataframe.values

#Preprocess the data with normalization since distribution is not gaussian and ranges are far different
scaler = MinMaxScaler() 
dataset = scaler.fit_transform(dataset)

#Cutoff is used to specify the amount of data that will be used for training
Cutoff = int(len(dataset)*.9)
Features = 8


#Training Features and Labels
TrainingInput = dataset[:Cutoff,0:Features]
TrainingOutput = dataset[:Cutoff,Features:]

#Testing Features and Labels
TestingInput = dataset[Cutoff:,0:Features]
TestingOutput = dataset[Cutoff:,Features:]

#For NN callback, stops when accuracy is not increasing
callback = EarlyStopping(monitor='accuracy', patience=3)

def CreateNNmodel(hp):
	#Function to create a neural network, set to be tuned with Hyperparameters
	model = Sequential()


	model.add(Dense(hp.Int("input_units", 8, 40, 2), input_dim=Features, activation=hp.Choice(
        'dense_activation',
        values=['relu', 'tanh', 'sigmoid'],default='relu'), kernel_initializer='he_normal'))
	model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_1', 0.0, 0.5, .05)))


	for i in range(hp.Int("n_layers", 1, 6)):
		model.add(Dense(hp.Int("dense_units", 8, 40, 2), activation=hp.Choice(
        'dense_activation',
        values=['relu', 'tanh', 'sigmoid'],default='relu'), kernel_initializer='he_normal'))
		model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_2', 0.0, 0.5, .05)))

	model.add(Dense(4, activation = 'softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def TestRandomForest():
	#Random forest testing model
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
	#Knn testing model
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
	#Neural network testing model
	model = CreateNNmodel()
	history = model.fit(TrainingInput, TrainingOutput, epochs=100,batch_size = Features, verbose=1,validation_data = (TestingInput, TestingOutput), callbacks = callback)
	acc = model.evaluate(TestingInput,TestingOutput)
	WOacc = withinOne(np.argmax(model.predict(TestingInput), axis=1), TestingOutput.argmax(axis=1))
	print("NN Accuracy: ", acc[1], "Within One: ", WOacc)
	NNAccuracies.append(acc[1])
	NNWOAccuracies.append(WOacc)
	results = np.argmax(model.predict(TestingInput), axis=1)
	cm = confusion_matrix(TestingOutput.argmax(axis=1), results) 
	return cm

def withinOne(prediction, actual):
	#Function that tests accuracy of an algorithims guess by decidiing if its within one of the category it belongs to
	acc = 0
	total = 0
	for p, a in zip(prediction,actual):
		if (p + 1 == a or p - 1 == a or p == a):
			acc +=1
		total +=1
	return acc/total

def ConfusionMatrix(cm, title):
	#Function that creates and output a confusion matrix
	categories = [ 'Above Average','Below Average','Blockbuster', 'Bust',]
	df_cm = pd.DataFrame(cm, index = categories, columns = categories)
	plt.figure(figsize = (10,7))
	plt.title(title)
	sns.heatmap(df_cm, annot=True, cmap = 'Blues')
	plt.show()


def TestModels():
	#Function that outputs confusion matrix of each of the algorithims tested
		ConfusionMatrix(TestKNN(), "Knn Confusion Matrix")
		ConfusionMatrix(TestRandomForest(), "Random Forest Confusion Matrix")
		ConfusionMatrix(TestNNModel(), "Neural Network Confusion Matrix")

def TuneNetwork():
	#Function to tune hyperparameters for the neural network
	tuner = Hyperband(
		CreateNNmodel,
		max_epochs=50,
		objective='val_accuracy',
		executions_per_trial=7,
		directory=os.path.normpath('C:/')
	)

	tuner.search(x=TrainingInput,y=TrainingOutput, validation_data= (TestingInput, TestingOutput),epochs = 50, batch_size = Features)
	best_model = tuner.get_best_models(num_models=1)[0]
	loss, accuracy = best_model.evaluate(TestingInput, TestingOutput)
	print(accuracy)










