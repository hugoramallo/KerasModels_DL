#step 1
import numpy as np
import pandas as pd
from keras.losses import mean_squared_error
from sklearn.model_selection import train_test_split

from regression import predictors_norm

#get file
concrete_data = pd.read_csv('https://cocl.us/concrete_data.csv')
#get data first rows
concrete_data.head()
#check datapoints
concrete_data.shape
#check the data
concrete_data.describe()
concrete_data.isnull().sum()
#get the columns from the concrete data
concrete_data_columns = concrete_data.columns
#get all the independent variables (predictors) except the target Strength
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength'] # Strength column
# Split the data with train_test_split, only a 30% of the data for the test
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)

#step 2
#Train the model on the training data using 50 epochs.

import keras
from keras.models import Sequential
from keras.layers import Dense

#build the neural network for regression model
def regression_model():
    # create model
    model = Sequential()
    # hidden layer 10 nodes
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    #out node
    model.add(Dense(1))
    # using optimizer adam and mean squared error
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#Create a model
model = regression_model()

#Training the model
model.fit(X_train, y_train, epochs=50, verbose=0)


#step 3
#Evaluate the model on the test data and compute the mean squared error
#between the predicted concrete strength and the actual concrete strength.

# Predict on the test data
y_pred = model.predict(X_test)

#Compute the mean squared error and add it to the list
mse = mean_squared_error(y_test, y_pred)
mse_list.append(mse)

#step 4

#Repeat 50 times
for i in range(50):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3)

    #Create the model
    model = regression_model()

    #Train the model
    model.fit(X_train, y_train, epochs=50, verbose=0)

    #Predict on the test data
    y_pred = model.predict(X_test)

    #Compute the mean squared error and add it to the list
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

    # Convert the list to a numpy array
    mse_array = np.array(mse_list)

    # Report the mean and standard deviation of the mean squared errors
    print(f"MSQ: {mse_array.mean()}")
    print(f"SD of the mean squared errors: {mse_array.std()}")