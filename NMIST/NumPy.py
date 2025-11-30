# https://www.kaggle.com/code/sunghyuneun/numpy-mnist
# Thanks Samson Zhang u the goat lowkey makes a lot more sense after doing this
# None of this code is runnable unless you run it on ran it in Kaggle link


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

data = np.array(data) #Converts (pandas) dataset into numpy array
m,n = data.shape #Extracts rows (m) and columns (n)
np.random.shuffle(data) #Shuffles data, prevents order bias

#First 1000 Rows are for development (testing)
data_dev = data[0:1000].T #Takes first 1000 rows of shuffled dataset, then Transposes. 
#Reason for T: Conventional to have features (pixels) as rows and examples as columns
#This means that dimensions went from m x n -> n x 1000
Y_dev = data_dev[0] #First Row of Transposed data (Labels)
X_dev = data_dev[1:n] # Remaining rows are Development Set Features
X_dev = X_dev / 255. #Normalized into values between 0-1

# Rest are for training
data_train = data[1000:m].T 
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape #Finds data as n * m_train

def init_params():
	W1 = np.random.rand(10,784) - 0.5 #rand creates array of shape of args, populated with random values from [0,1)
	b1 = np.random.rand(10,1) - 0.5 # -0.5 obv makes it [-0.5, 0.5)
	W2 = np.random.rand(10,10) - 0.5
	b2 = np.random.rand(10,1) - 0.5
	return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z) #simple ReLU Function

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z)) #Softmax function
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 #Forward propagation - see equations above
    A1 = ReLU(Z1)
    Z2 = W2.dot(A2) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def ReLU_prime(Z):
    return Z > 0 #If Z it is bigger than 0, then return 1. Otherwise return 0

def one_hot(Y): #one-hot encoding; converting categorical data to a numerical format for ML
	#You create a binary column for each category, 1 indicates that category exists
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) #Number of data, number of possible data
    one_hot_Y[np.arange(Y.size), Y] = 1 #np.arange generates indices for the row, Y is the column index (actual digit). Sets element at row index, column index to 1.
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_prime(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def param_update(W1, b1, W2, b2, alpha, dW1, db1, dW2, db2):
    W1 -= alpha * dW1 #Parameter update, see equations above.
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2
    
def get_predictions(A2):
    return np.argmax(A2, 0)  #Takes in result, and finds the index of the biggest probability

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size #Average number of correct predictions

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = param_update(W1, b1, W2, b2, alpha, dW1, db1, dW2, db2)

        if (i % 10 == 0):
            print(f"Iteration: {i}")
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500) #Literally trains, with .1 learning rate and 500 iterations

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X) #Finds result
    predictions = get_predictions(A2) #Gets Prediction
    return predictions

def test_prediction(index, W1, b1, W2, b2): #tests on 1 specific image
    current_image = X_train[:, index, None] # takes 1 column (image) and adds an axis (None)
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
test_prediction(0, W1, b1, W2, b2)
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)
