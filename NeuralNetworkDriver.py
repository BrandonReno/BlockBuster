import tensorflow as tf
import pandas as pd
import numpy as np
import os.path
from os import path
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self):
        self.Data = self.setData()
        self.Output = []
        self.SplitData()
        #Training Data
        self.trainingData = np.array(self.Data[:3080])
        self.trainingDataY = np.array(self.Output[:3080])
        #Testing Data
        self.testingData = self.Data[3080:]
        self.testingDataY = self.Output[3080:]

        self.trainingData = tf.keras.utils.normalize(self.trainingData, axis = 1)
        self.testingData = tf.keras.utils.normalize(self.testingData, axis = 1)



    def setData(self):
        data = []
        if path.exists("CleanedDataRevenue.csv"):
            #I can make cleaned data a variable and then this function would look to see what that file is called in data driver again something for later
            with open("CleanedDataRevenue.csv") as f:
                next(f)
                for line in f:
                    data.append([float(x) for x in line.split(",")])
            return data
        else:
            pass
            #I am going to create the file and then set the data recursively for now I will pass this
            #This is not a huge priority in the end just more so for if different data files are used
    def SplitData(self):
        for arr in self.Data:
            test = []
            for i in range(1,11):
                if i == arr[5]:
                    test.append(1)
                else:
                    test.append(0)
            self.Output.append(arr[5])
            arr.pop(5)

    def CreateRegressionNetwork(self):
        model = tf.keras.models.Sequential()
        #Input layer takes  in input size of 5 and then first hidden layer with 12 nodes
        model.add(tf.keras.layers.Dense(12, input_dim = 5, kernel_initializer='normal', activation='relu'))
        #Second hidden layer has 8 nodes
        model.add(tf.keras.layers.Dense(8, kernel_initializer='normal', activation = 'relu'))
        #Output layer 
        model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))
        #compile the model
        model.compile(loss='mean_squared_error', optimizer = 'adam')

        return model

    def CreateClassificationNetwork(self):
        model = tf.keras.models.Sequential()
        #Input layer takes  in input size of 5 and then first hidden layer with 12 nodes
        model.add(tf.keras.layers.Dense(12, input_dim = 5, activation='relu'))
        #Second hidden layer has 8 nodes
        model.add(tf.keras.layers.Dense(8, activation = 'relu'))
        #Output layer 
        model.add(tf.keras.layers.Dense(1, activation= 'sigmoid'))



if __name__ == "__main__":
    n = NeuralNetwork()
    kfold = KFold(n_splits=10)
    Network = n.CreateRegressionNetwork()
    
    Network.fit(n.trainingData, n.trainingDataY, epochs=100,validation_split=.1, batch_size=5, verbose=0)
    #results = cross_val_score(estimator, n.trainingData, n.trainingDataY, cv=kfold)
    #print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
   
    

  