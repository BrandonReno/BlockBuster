import pandas
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import normalize
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# load dataset
dataframe = pandas.read_csv("CleanedData.csv", header=0)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:3000,0:5]
Y = dataset[:3000,5]

XT = dataset[3000:,0:5]
YT = dataset[3000:,5]


callback = EarlyStopping(monitor='mse', patience=3)

def Linear_Regression_Model():
	pass


def Regr_model():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(18, activation='relu'))
	model.add(Dense(12, activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mse', optimizer='adam', metrics=['mse'])
	return model

model = Regr_model()
model.fit(X, Y, epochs = 200, verbose =2, callbacks=callback)
print(model.evaluate(X,Y, verbose=0))


history = model.fit(X, Y, epochs=200, verbose=2, callbacks=callback)

plt.plot(history.history['mse'])
plt.title("Mean Squared Error Movie Revenue")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()





