import pandas as pd

path = "/content/drive/MyDrive/NFLX.csv"
df = pd.read_csv(path)
df = df[df['Close'] != 0]   #removing all zero values

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math

figure(figsize=(20,10),dpi=80)
time_x = df.index
time_y = df['Close']
plt.plot(time_x,time_y)
# plt.show()

# ! pip install EMD-signal

Signal = df['Close'].to_numpy()
T = df.index.to_numpy()
from PyEMD import EMD
emd = EMD()
IMFs = emd(Signal)
nIMFs = len(IMFs)

# plt.figure(figsize=(12,9))
# plt.subplot(nIMFs+10, 1, 1)
# plt.plot(T, Signal, 'r')

for n in range(nIMFs):
  plt.figure(figsize=(12,9))
  plt.subplot(nIMFs+1, 1,2)
  plt.plot(T, IMFs[n], 'g')
  plt.ylabel("IMF %i" %(n+1))
  plt.locator_params(axis='y', nbins=5)


# plt.xlabel("Time")
# plt.show()

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_dataset(dataset, look_back):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def perform_LSTM(dataset, look_back, layer=4):
  
  dataset = dataset.astype('float32')
  dataset = np.reshape(dataset, (-1, 1))
  
  # Normalize the data -- using Min and Max values in each subsequence to normalize the values
  scaler = MinMaxScaler()
  dataset = scaler.fit_transform(dataset)
  
  # Split data into training and testing set
  train_size = int(len(dataset) * 0.9)
  test_size = len(dataset) - train_size
  train, test = dataset[0:train_size, :], dataset[train_size:, :]
  
  trainX, trainY = create_dataset(train, look_back)
  testX, testY = create_dataset(test, look_back)

  trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
  testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
  
  # create and fit the LSTM network
  model = Sequential()
  model.add(LSTM(layer, input_shape=(1, look_back)))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

  # make predictions
  trainPredict = model.predict(trainX)
  testPredict = model.predict(testX)

  # invert predictions
  trainPredict = scaler.inverse_transform(trainPredict)
  trainY = scaler.inverse_transform([trainY])
  testPredict = scaler.inverse_transform(testPredict)
  testY = scaler.inverse_transform([testY])
  testing_error = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
  return testPredict, testY, testing_error

IMF_predict_list = []
error_list = []
for IMF in IMFs:
  IMF_predict, IMF_test, testing_error = perform_LSTM(IMF, 10, layer=4)
  error_list.append(testing_error)
  IMF_predict_list.append(IMF_predict)

final_prediction = []
for i in range(len(IMF_predict_list[0])):
  element = 0 
  for j in range(len(IMF_predict_list)):
    element += IMF_predict_list[j][i]
  final_prediction = final_prediction + element.tolist()

SP = time_y.astype('float32')
SP = np.reshape(SP.to_numpy(), (-1, 1))

train_size = int(len(SP) * 0.9)
test_size = len(SP) - train_size
SP_train, SP_test = SP[0:train_size], SP[train_size:]

SP_testX, SP_testY = create_dataset(SP_test, 10)

math.sqrt(mean_squared_error(SP_testY.tolist(), final_prediction))

figure(figsize=(10, 8), dpi=80)
x = np.linspace(1, len(final_prediction)+1, len(final_prediction), endpoint=True)
# plot lines
plt.plot(x, final_prediction, label = "Predicted Value")
plt.plot(x, SP_testY.tolist(), label = "Actual Value")
plt.legend()
plt.show()