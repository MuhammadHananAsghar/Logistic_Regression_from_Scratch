"""
@author: Muhammad Hanan Asghar
"""

# LOGISTIC REGRESSION FROM SCRATCH

# Libraries
from sklearn.datasets import make_classification
import numpy as np


# Sigmoid Function aka Activation Function
def sigmoid(z):
    # z: Output From Logistic Regression Equation
    
    return 1.0 / (1 + np.exp(-z))

# Log Loss
def loss(y, a):
    # y: True Output
    # a: Output From Sigmoid
    
    loss = -np.mean(y*np.log(a) - (1-y)*np.log(1-a))
    return loss

def normalize(X):
    
    # X --> Input.
    
    # m-> number of training examples
    # n-> number of features 
    m, n = X.shape
    
    # Normalizing all the n features of X.
    for i in range(n):
        X = (X - X.mean(axis=0))/X.std(axis=0)
        
    return X

# Random Classification Data
X, y = make_classification(n_samples=100000, n_features=5, n_redundant=0, 
                           n_informative=2, random_state=1, 
                           n_clusters_per_class=1)

m, n = X.shape
# Reshaping y.
y = y.reshape(m,1)
# Normalizing
x = normalize(X)
# Batch Size
bs = 100
# Epoch
EPOCHS = 3000

    
# TRAINING FUNCTION

def fit(epochs, bs, x, y):
    
    # Initial Weights
    m, n = x.shape
    w = np.zeros((n, 1))
    b = 0
  
    # Loss Variable
    J = 0
  
    # Weights Derivative
    dw = 0
  
    # Bias Derivative
    db = 0
  
    # Losses
    losses = []

    # Learning Rate
    alpha = 0.01

    # SGD = Stochastic Gradient Descent
    for epoch in range(epochs):
        print(f"Epoch: {epoch} ---> ")
      # for i in tqdm(range(m // bs), desc=f'Epoch: {epoch}'):
        for i in range(m // bs):
            start = i * bs
            end = start + bs

            # Getting Batch of Size 'bs'       
            x_batch = x[start:end]
            y_batch = y[start:end]

            # Logistic Regression Equation
            Z = np.dot(x_batch, w) + b

            # Calculating Sigmoid
            A = sigmoid(Z)

            # Calculating Gradients
            m, n = x_batch.shape
            dz = A - y_batch
            dw = (1/m) * np.dot(x_batch.T, dz)
            db = (1/m) * np.sum(dz)

            # Updating Gradients
            w = w - alpha*dw
            b = b - alpha*db

        # Losses
        J = loss(y, sigmoid(np.dot(x, w) + b))
        losses.append(J)
    
    return w, b, np.sum(losses)/len(losses)

weights, bias, mn_loss = fit(EPOCHS, bs, x, y)

# PREDICTING FUNCTION
def predict(x, weights, bias):
    inp = normalize(x)

    # Logistic Regression Equation
    Z = np.dot(inp, weights) + bias

    # Calculating Sigmoid
    preds = sigmoid(Z)

    pred_class = []
    pred_class = [1 if i > 0.5 else 0 for i in preds]

    return pred_class

# Random Classification Data
X, y = make_classification(n_samples=10, n_features=5, n_redundant=0, 
                           n_informative=2, random_state=1, 
                           n_clusters_per_class=1)

pred = predict(X, weights, bias)

print(pred, y)
# THANK YOU