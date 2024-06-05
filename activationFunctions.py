import torch.nn as nn
import torch

import matplotlib.pyplot as plt
torch.manual_seed(2)


#LOGISTIC FUNCTION  
# Create a tensor

z = torch.arange(-10, 10, 0.1,).view(-1, 1)

# Create a sigmoid object

sig = nn.Sigmoid()

# Make a prediction of sigmoid function

yhat = sig(z)

# Plot the result

plt.plot(z.detach().numpy(),yhat.detach().numpy())
plt.xlabel('z')
plt.ylabel('yhat')
plt.title('sigmoid function')
plt.show()

# Use the build in function to predict the result

yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')
plt.title('sigmoid function')
plt.show()

#tanH FUNCTION
# Create a tanh object

TANH = nn.Tanh()

# Make the prediction using tanh object

yhat = TANH(z)
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')
plt.title('tanh function')
plt.show()

# Make the prediction using the build-in tanh object

yhat = torch.tanh(z)
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')
plt.title('tanh function')
plt.show()

#RELU FUNCTION

# Create a relu object and make the prediction

RELU = nn.ReLU()
yhat = RELU(z)
plt.plot(z.numpy(), yhat.numpy())

# Use the build-in function to make the prediction

yhat = torch.relu(z)
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')
plt.title('relu function')
plt.show()

# Plot the results to compare the activation functions

x = torch.arange(-2, 2, 0.1).view(-1, 1)
plt.plot(x.numpy(), torch.relu(x).numpy(), label='relu')
plt.plot(x.numpy(), torch.sigmoid(x).numpy(), label='sigmoid')
plt.plot(x.numpy(), torch.tanh(x).numpy(), label='tanh')
plt.legend()
plt.title('Activation Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.show() 

print("The ReLU function is the most simple one, thresholding on zero. The sigmoid function squashes the input for each element between 0 and 1. The tanh rescales the input for each element between -1 and 1.")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>end of activationFunctions.py<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")