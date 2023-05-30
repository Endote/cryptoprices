from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import pandas as pd
import requests as req
import tensorflow.compat.v2 as tf
import time


class Binance:

    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret

        if load_model('model.h5') is None:
            print('Model being built')
            # self.model = self.build_model()
            pass
        else:
            print('Model loaded')
            self.model = load_model('model.h5')
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
    def create_dataset(self, dataset, look_back=10):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    def build_model(data, look_back=10):
        # Split the data into training and test datasets
        data_X, data_Y = client.create_dataset(data, look_back)
        X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.2, shuffle=False)

# Reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

### MODEL

# Create the LSTM model
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(1, look_back)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

# Compile the model
        self.model.compile(loss='mean_squared_error', optimizer='adam')


# Train the model
        self.model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=1)
        self.model.fit(X_train, Y_train, epochs=100, verbose=1)



# Make predictions
        train_predict = self.model.predict(X_train)
        test_predict = self.model.predict(X_test)

# Invert predictions to get the actual price
        train_predict = scaler.inverse_transform(train_predict)
        Y_train = scaler.inverse_transform([Y_train])

        test_predict = scaler.inverse_transform(test_predict)
        Y_test = scaler.inverse_transform([Y_test])

        self.model.save('model_'+look_back+'.h5')

if __name__ == '__main__':
# Define how many previous days of data we want to use to predict the next data point


# Enter your Binance API key and API secret here
    api_key = '7ADjwVTbIBk9LZHWuDTjJJncrzkCfWzygsnUDJYKYkVoghyNaUYkY7qajWasK3fj'
    api_secret = 'xm9ViUOA2Yw8KAZl8yIg9UxboDYrNjsoQv1qqcNHQK8kSsIXCeRQ5gFx2gkPji6j'

    client = Binance(api_key, api_secret)

    client.get_tokens()

    scaler = MinMaxScaler()

# data_close = client.tokens[2]['Close'].values.reshape(-1,1)

    data_close = client.tokens[2]['Close'].values
    data = scaler.fit_transform(data_close.reshape(-1, 1))

    self.build_model(data)




# Print the predictions
    # print('Train Predicted:', train_predict[-1])
    print('Test Predicted:', test_predict)

    print(len(test_predict))

    print('ACTUAL\n\n\n')
    print(client.tokens[2][-519:])






