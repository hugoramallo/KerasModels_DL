import pandas as pd
import numpy as np
concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()
#check datapoints
concrete_data.shape
#check the data
concrete_data.describe()
concrete_data.isnull().sum()

#divididimos los datos, predictors (variables independientes), target (variable indepdendiente)
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

#checkeamos los predictores
predictors.head()
target.head()

#normalizamos los datos
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

#guardamos los predictores en nuestra variable
n_cols = predictors_norm.shape[1] # number of predictors

import keras
from keras.models import Sequential
from keras.layers import Dense

#método para construir la red neuronal
# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#creamos nuestro modelo
model = regression_model()

#entrenamos nuestro modelo con 100 pasos
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)