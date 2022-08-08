# improt the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN


# lets create a sine wave data
x = range(75)
X = [np.sin(2.0 * np.pi * i / 30) for i in x ]

# adding a noise in sine wave data
XN = X+0.1*np.random.uniform(low=-1.0, high=1.0, size=len(x))

int(len(XN))


# lets plot the sine wave without noise and without noise
plt.figure(figsize=(8,5))
plt.plot(X, linewidth=4)
plt.title("Sine Wave Without Noise", fontsize=15)
plt.show()

# lets make a dataframe of sine wave with noise data
df1 = pd.DataFrame(XN).values

plt.figure(figsize=(8,5))
plt.plot(df1, linewidth=4, color='red')
# plt.plot(range(0,int(len(XN)/2+1)),df1[:int(len(XN)/2+1)], linewidth=4)
# plt.plot(range(int(len(XN)/2),int(len(XN))),df1[int(len(XN)/2):], linewidth=4)
plt.title("Sine Wave With Noise", fontsize=15)
# plt.legend()
plt.show()

# lets check the sine wave with noise data valies
print(df1)


firstHalfCyclePercentage = (15/len(X))
firstHalfCyclePercentage


# splitting dataset into 20% training and 80% testing split
training_size=int(len(df1)*0.20)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

print("Training Size : ",training_size)
print("Testing Size : ",test_size)


# lets make a dataset for time series
def make_dataset_for_time_series_analysis(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0] 
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)



# predict the future values on how many previous values
previous_values = 2


time_step = previous_values
X_train, y_train = make_dataset_for_time_series_analysis(train_data, time_step)
X_test, ytest = make_dataset_for_time_series_analysis(test_data, time_step)

# lets reshape the training and testing data that actually use for RNN model
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# Create the RNN model with 30 hidden neurons
model = Sequential()
model.add(SimpleRNN(units = 30, input_shape=(None,1)))
# Adding the output layer
model.add(Dense(units = 1))
# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
# check the summary of model
model.summary()

#fit the model
import time
start = time.time()
# lets fit the model
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=500,batch_size=4,verbose=1)
end = time.time()
print(f"Execution time: {end-start} seconds")


# lets check the predicted values on training and testing
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


plt.figure(figsize=(8,5))
look_back=2

# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

# plot baseline and predictions
plt.plot(range(0,16),df1[:16], label='Original', linewidth=4)
plt.plot(testPredictPlot, label='Prediction', linewidth=4,color='black')
plt.legend()
plt.show()

