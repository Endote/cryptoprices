from binance.api import API
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import binance
import numpy as np
import os
import pandas as pd
import requests as req
import tensorflow.compat.v2 as tf
import time


class Binance:

    def __init__(self, api_key, api_secret):
        # Define how many previous days of data we want to use to predict the next data point
        self.look_back = 100
        self.api_key = api_key
        self.api_secret = api_secret
        self.model = load_model('model.h5')
        if self.model is None:
            self.model = Sequential()
            self.model.add(LSTM(50, input_shape=(1, self.look_back)))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1))
        print('constructed')

    def exchange_info(self):
        print('text')
        r = req.get('https://api.binance.com/api/v3/exchangeInfo')

        data = r.json()
        with open('json\\'+str(time.time())[0:10]+'.json', 'a') as f:
            f.write(r.text)
        
        return data

    def get_tokens(self):

        self.tokens = []

# traverse root directory, and list directories as dirs and files as files
        for root, dirs, files in os.walk('data'):
            path = root.split(os.sep)
            print((len(path) - 1) * '---', os.path.basename(root))
            for file in files:
                print(len(path) * '---', file)
                if 'csv' in file:
                    tmp_path = root+ '\\'+ file
                    print(tmp_path)
                    self.tokens.append(pd.read_csv(tmp_path).drop(columns=['SNo','Name','High','Low', 'Symbol']))      

        print(self.tokens)

        # Function to create dataset
    def create_dataset(self, dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

### DATA




# Enter your Binance API key and API secret here
api_key = '7ADjwVTbIBk9LZHWuDTjJJncrzkCfWzygsnUDJYKYkVoghyNaUYkY7qajWasK3fj'
api_secret = 'xm9ViUOA2Yw8KAZl8yIg9UxboDYrNjsoQv1qqcNHQK8kSsIXCeRQ5gFx2gkPji6j'

client = Binance(api_key, api_secret)

client.get_tokens()

scaler = MinMaxScaler()

# data_close = client.tokens[2]['Close'].values.reshape(-1,1)

data_close = client.tokens[2]['Close'].values
data = scaler.fit_transform(data_close.reshape(-1, 1))


# Split the data into training and test datasets
data_X, data_Y = client.create_dataset(data, look_back)
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

print(X_train)
print(Y_train)


### MODEL

# Create the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')


# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=1)


# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions to get the actual price
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])

test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# Print the predictions
print('Train Predicted:', train_predict)
print('Test Predicted:', test_predict)






