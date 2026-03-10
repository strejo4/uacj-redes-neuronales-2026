# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.optimizers import SGD
from keras import activations
import math
from sklearn.metrics import mean_squared_error
    
#%%
# get the data
path_file = 'D:/UACJ D/MIAAD/Redes Neuronales Profundas/DataSets/IBM_2006-01-01_to_2018-01-01.csv'
dataset1 = pd.read_csv(path_file, index_col='Date', parse_dates=['Date'])
dataset1.head()  

dataset = dataset1.dropna()  # Eliminar cualquier fila con NaN en cualquier columna

training_set = dataset.loc[:'2016', ["Open"]].values  # modificar dependiendo de la tarea
test_set = dataset.loc['2017':, ["Open"]].values   # modificar dependiendo de la tarea


#%%
# We have chosen 'High' attribute for prices. Let's see what it looks like
dataset["Low"][:'2016'].plot(figsize=(16,4),legend=True)
dataset["Low"]['2017':].plot(figsize=(16,4),legend=True)
plt.legend(['Training set (Before 2017)','Test set (2017 and beyond)'])
plt.title('IBM stock price')
plt.show()

#%%
# Scaling the training set
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# we create a data structure with 60 timesteps and 1 output
# So for each element of training set, we have 60 previous training set elements 
X_train = []
y_train = []
previous_days = 30  # cantidad de días previos usados para la predicción
N = training_set.shape[0]
for i in range(previous_days,N):
    X_train.append(training_set_scaled[i-previous_days:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)


#%%
# Reshaping X_train 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))    

#%% 

# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1, activation='linear'))

# Compiling the RNN
regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
regressor.summary()
#%%
# Fitting to the training set
regressor.fit(X_train,y_train,epochs=10,batch_size=32)


#%%
# get the test set ready in a similar way as the training set.
dataset_total = pd.concat((dataset["High"][:'2016'],dataset["High"]['2017':]),axis=0)
inputs = dataset_total[len(dataset_total)-len(test_set) - previous_days:].values
inputs = inputs.reshape(-1,1)
inputs  = sc.transform(inputs)

# Preparing X_test and predicting the prices
X_test = []
N1 = inputs.shape[0]
for i in range(previous_days,N1):
    X_test.append(inputs[i-previous_days:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

#%%
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#%%
# Visualizing the results for LSTM

plt.plot(test_set, color='red',label='Real IBM Stock Price')
plt.plot(predicted_stock_price, color='blue',label='Predicted IBM Stock Price')
plt.title('IBM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('IBM Stock Price')
plt.legend()
plt.show()
# Evaluating our model
rmse = math.sqrt(mean_squared_error(test_set, predicted_stock_price))
print("The root mean squared error is {}.".format(rmse))