import torch.nn as nn
import torch

import matplotlib.pyplot as plt
torch.manual_seed(2)

#logistic activation function
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
plt.title('sigmoid function'
          )

# Use the build in function to predict the result

yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('sigmoid function')
plt.show()

# Create a tanh object

TANH = nn.Tanh()
# Make the prediction using tanh object

yhat = TANH(z)
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')
plt.title('tanh function')
plt.show()
#For custom modules, call the Tanh object from the torch (nn.functional for the old version), #which applies the element-wise sigmoid from the function module and plots the results:
# Make the prediction using the build-in tanh object

yhat = torch.tanh(z)
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')
plt.title('tanh function')
plt.show()

#relu
# Create a relu object and make the prediction

RELU = nn.ReLU()
yhat = RELU(z)
plt.plot(z.numpy(), yhat.numpy())

# Use the build-in function to make the prediction

yhat = torch.relu(z)
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')
plt.title('Relu function')
plt.show()

# Plot the results to compare the activation functions

x = torch.arange(-2, 2, 0.1).view(-1, 1)
plt.plot(x.numpy(), torch.relu(x).numpy(), label='relu')
plt.plot(x.numpy(), torch.sigmoid(x).numpy(), label='sigmoid')
plt.plot(x.numpy(), torch.tanh(x).numpy(), label='tanh')
plt.xlabel('z')
plt.ylabel('yhat')
plt.title('activation functions')
plt.legend()


x = torch.arange(-1, .1, 0.1).view(-1, 1)
plt.plot(x.numpy(), torch.relu(x).numpy(), label='relu')
plt.plot(x.numpy(), torch.sigmoid(x).numpy(), label='sigmoid')
plt.plot(x.numpy(), torch.tanh(x).numpy(), label='tanh')
plt.title('activation functions with negative values')
plt.legend()

print(">>>>>>>>>>>>>>>>>End of the NN activation function implementation<<<<<<<<<<<<<<<<<<<<<")