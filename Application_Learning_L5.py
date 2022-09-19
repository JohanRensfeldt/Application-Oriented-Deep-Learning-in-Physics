# %% [markdown]
# # Exercise 5.1
# 

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# **Simple Network**
# 
# We continue with the dataset first encountered in the previous exercise. Please refer to the discussion there for an introduction to the data and the learning objective.
# 
# Here, we manually implement a simple network architecture

# %%
# The code snippet below is responsible for downloading the dataset
# - for example when running via Google Colab.
#
# You can also directly download the file using the link if you work
# with a local setup (in that case, ignore the !wget)

#!wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

# %%
# Before working with the data, 
# we download and prepare all features

# load all examples from the file
data = np.genfromtxt('winequality-white.csv',delimiter=";",skip_header=1)

print("data:", data.shape)

# Prepare for proper training
np.random.shuffle(data) # randomly sort examples

# take the first 3000 examples for training
# (remember array slicing from last week)
X_train = data[:3000,:11] # all features except last column
y_train = data[:3000,11]  # quality column

# and the remaining examples for testing
X_test = data[3000:,:11] # all features except last column
y_test = data[3000:,11] # quality column

print("First example:")
print("Features:", X_train[0])
print("Quality:", y_train[0])

# %% [markdown]
# # Problems
# 
# The goal is to implement the training of a neural network with one input layer, one hidden layer, and one output layer using gradient descent. We first (below) define the matrices and initialise with random values. We need W, b, W' and b'. The shapes will be:
#   * W: (number of hidden nodes, number of inputs) named `W`
#   * b: (number of hidden nodes) named `b`
#   * W': (number of hidden nodes) named `Wp`
#   * b': (one) named `bp`
# 
# Your tasks are:     
#    * Implement a forward pass of the network as `dnn` (see below)
#    * Implement a function that uses one data point to update the weights using gradient descent. You can follow the `update_weights` skeleton below
#    * Now you can use the code below (training loop and evaluation) to train the network for multiple data points and even over several epochs. Try to find a set of hyperparameters (number of nodes in the hidden layer, learning rate, number of training epochs) that gives stable results. What is the best result (as measured by the loss on the training sample) you can get?

# %%
# Initialise weights with suitable random distributions
hidden_nodes = 50 # number of nodes in the hidden layer
n_inputs = 11 # input features in the dataset

W = np.random.randn(hidden_nodes,11)*np.sqrt(2./n_inputs)
b = np.random.randn(hidden_nodes)*np.sqrt(2./n_inputs)
Wp = np.random.randn(hidden_nodes)*np.sqrt(2./hidden_nodes)
bp = np.random.randn((1))

print(W.shape)
print(b.shape)
print(Wp.shape)

# %%
# You can use this implementation of the ReLu activation function
def relu(x):
    return np.maximum(x, 0)

# %%
def dnn(x,W,b,Wp,bp):
    y = Wp*relu(np.dot(W,x)+b)+bp
    return y

# %%
def update_weights(x,y, W, b, Wp, bp):
    
    learning_rate = 0.01

    out = dnn(x,W,b,Wp,bp)


    bp_prime = 2*(out-y)

    b_prime = 2*(out-y)*Wp* np.heaviside(np.dot(W,x) + b,0)

    Wp_prime = 2*(out-y)*relu(np.dot(W,x) + b)
    
    Wk_prime = np.outer(2*(out-y)*Wp * np.heaviside(np.dot(W,x) + b,0), x) 

    heavy  =  np.heaviside(np.dot(W,x) + b,0)

    W_new = W - learning_rate * Wk_prime
    b_new = b - learning_rate * b_prime
    Wp_new = Wp - learning_rate * Wp_prime
    bp_new = bp - learning_rate * bp_prime

    print("W_new", W_new.shape)
    print("b_new", b_new.shape)
    print("Wp_new", Wp_new.shape)
    print("bp_new", bp_new.shape)



    # TODO: Derive the gradient for each of W,b,Wp,bp by taking the partial
    # derivative of the loss function with respect to the variable and
    # then implement the resulting weight-update procedure
    # See Hint 2 for additional information

    # You might need these numpy functions:
    # np.dot, np.outer, np.heaviside
    # Hint: Use .shape and print statements to make sure all operations
    # do what you want them to 
    
    # TODO: Update the weights/bias following the rule:  weight_new = weight_old - learning_rate * gradient

    


    return W_new, b_new, Wp_new, bp_new # return the new weights

# %% [markdown]
# # Training loop and evaluation below

# %%
# The code below implements the training.
# If you correctly implement  dnn and update_weights above, 
# you should not need to change anything below. 
# (apart from increasing the number of epochs)

train_losses = []
test_losses = []

# How many epochs to train
# This will just train for one epoch
# You will want a higher number once everything works
n_epochs = 1

# Loop over the epochs
for ep in range(n_epochs):
        
    # Each epoch is a complete over the training data
    for i in range(X_train.shape[0]):
        
        # pick one example
        x = X_train[i]
        y = y_train[i]

        # use it to update the weights
        W,b,Wp,bp = update_weights(x,y,W,b,Wp,bp)
    
    # Calculate predictions for the full training and testing sample
    y_pred_train = [dnn(x,W,b,Wp,bp)[0] for x in X_train]
    y_pred = [dnn(x,W,b,Wp,bp)[0] for x in X_test]

    # Calculate aver loss / example over the epoch
    train_loss = sum((y_pred_train-y_train)**2) / y_train.shape[0]
    test_loss = sum((y_pred-y_test)**2) / y_test.shape[0] 
    
    # print some information
    print("Epoch:",ep, "Train Loss:", train_loss, "Test Loss:", test_loss)
    
    # and store the losses for later use
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    
# After the training:
    
# Prepare scatter plot
y_pred = [dnn(x,W,b,Wp,bp)[0] for x in X_test]

print("Best loss:", min(test_losses), "Final loss:", test_losses[-1])

print("Correlation coefficient:", np.corrcoef(y_pred,y_test)[0,1])
plt.scatter(y_pred_train,y_train)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Prepare and loss over time
plt.plot(train_losses,label="train")
plt.plot(test_losses,label="test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# %% [markdown]
# # Hint 1

# %% [markdown]
# We want a network with one hidden layer. As activiation in the hidden layer $\sigma$ we apply element-wise ReLu, while no activation is used for the output layer. The forward pass of the network then reads:
# $$\hat{y}=\mathbf{W}^{\prime} \sigma(\mathbf{W} \vec{x}+\vec{b})+b^{\prime}$$

# %% [markdown]
# # Hint 2

# %% [markdown]
# For the regression problem the objective function is the mean squared error between the prediction and the true label $y$: 
# $$
# L=(\hat{y}-y)^{2}
# $$
# 
# Taking the partial derivatives - and diligently the applying chain rule - with respect to the different objects yields:
# 
# $$
# \begin{aligned}
# \frac{\partial L}{\partial b^{\prime}} &=2(\hat{y}-y) \\
# \frac{\partial L}{\partial b_{k}} &=2(\hat{y}-y) \mathbf{W}_{k}^{\prime} \theta\left(\sum_{i} \mathbf{W}_{i k} x_{i}+b_{k}\right) \\
# \frac{\partial L}{\partial \mathbf{W}_{k}^{\prime}} &=2(\hat{y}-y) \sigma\left(\sum_{i} \mathbf{W}_{i k} x_{i}+b_{k}\right) \\
# \frac{\partial L}{\partial \mathbf{W}_{k m}} &=2(\hat{y}-y) \mathbf{W}_{m}^{\prime} \theta\left(\sum_{i} \mathbf{W}_{i m} x_{i}+b_{m}\right) x_{k}
# \end{aligned}
# $$
# 
# Here, $\Theta$ denotes the Heaviside step function.


