import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchsummary import summary
from PIL import Image
import deltacnn

# Preprocessing data: convert to tensors and normalize by subtracting dataset
# mean and dividing by std.
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

# Get data from torchvision.datasets
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Define data loaders used to iterate through dataset
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# Show some example images
fig, axs = plt.subplots(5, 5, figsize=(5, 5))
for i in range(25):
    x, _ = test_data[i]
    ax = axs[i // 5][i % 5]
    ax.imshow(x.view(28, 28), cmap='gray')
    ax.axis('off')
    ax.axis('off')
plt.tight_layout()
plt.show()

def try_gpu():
    """
    If GPU is available, return torch.device as cuda:0; else return torch.device
    as cpu.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

def evaluate_accuracy(data_loader, net, device=torch.device('cpu')):
    """Evaluate accuracy of a model on the given data set."""
    net.eval()  #make sure network is in evaluation mode

    #init
    acc_sum = torch.tensor([0], dtype=torch.float32, device=device)
    n = 0

    for X, y in data_loader:
        # Copy the data to device.
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))
            n += y.shape[0] #increases with the number of samples in the batch
    return acc_sum.item()/n

class Linear(object):
    """
    Fully connected layer.
    
    Args:
        in_features: number of input features
        out_features: number of output features
    """

    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        # Define placeholder tensors for layer weight and bias. The placeholder
        # tensors should have the correct dimension according to the in_features
        # and out_features variables.
        self.weight = torch.Tensor(in_features, out_features)
        self.bias = torch.Tensor(out_features)

        # Initialize parameters
        self.init_params()

        # Define a cache varible to save computation, because some of the
        # forward pass values would be used during backward pass.
        self.cache = None

        # Define variables to store the gradients of the weight and bias
        # calculated during the backward pass
        self.weight_grad = None
        self.bias_grad = None

    def init_params(self, std=1.):
        """
        Initialize layer parameters. Sample weight from Gaussian distribution
        and bias uniform distribution.
        
        Args:
            std: Standard deviation of Gaussian distribution (default: 1.0)
        """

        self.weight = std*torch.randn_like(self.weight)
        self.bias = torch.rand_like(self.bias)

    def forward(self, x):
        """
        Forward pass of linear layer: multiply input tensor by weights and add
        bias. Store input tensor as cache variable.
        
        Args:
            x: input tensor with shape of (N, d1, d2, ...) where
                d1 x d2 x ... = in_features

        Returns:
            y: output tensor with shape of (N, out_features)
        """

        x = x.view(x.shape[0], -1)
        y = torch.mm(x, self.weight) + self.bias
        self.cache = x

        return y

    def backward(self, dupstream):
        """
        Backward pass of linear layer: calculate gradients of loss with respect
        to weight and bias and return downstream gradient dx.
        
        Args:
            dupstream: Gradient of loss with respect to output of this layer.

        Returns:
            dx: Gradient of loss with respect to input of this layer.
        """

        x = self.cache
        dx = torch.mm(dupstream, self.weight.T)
        self.weight_grad = torch.mm(x.T, dupstream)
        self.bias_grad = torch.sum(dupstream, dim=0)

        return dx

class ReLU(object):
    """
    ReLU non-linear activation function.
    """

    def __init__(self):
        super(ReLU, self).__init__()

        # Define a cache varible to save computation, because some of the
        # forward pass values would be used during backward pass.
        self.cache = None

    def forward(self, x):
        """
        Forward pass of ReLU non-linear activation function: y=max(0, x).
        
        Args:
            x: input tensor

        Returns:
            y: output tensor
        """

        y = torch.clamp(x, min=0)  # forward pass
        self.cache = x

        return y

    def backward(self, dupstream):
        """
        Backward pass of ReLU non-linear activation function: return downstream
        gradient dx.
        
        Args:
            dupstream: Gradient of loss with respect to output of this layer.

        Returns:
            dx: Gradient of loss with respect to input of this layer.
        """

        # Making sure that we don't modify the incoming upstream gradient
        dupstream = dupstream.clone()

        x = self.cache
        dx = dupstream
        dx[x < 0] = 0

        return dx

class Net(deltacnn.DCModule):
    """
    3-layer CNN network with max pooling
    
    Args:
        in_channels: number of features of the input image ("depth of image")
        hidden_channels: number of hidden features ("depth of convolved images")
        out_features: number of features in output layer
    """
    
    def __init__(self, in_channels, hidden_channels, out_features):
        super(Net, self).__init__()
        self.sparsify = deltacnn.DCSparsify()
        self.conv1 = deltacnn.DCConv2d(in_channels, hidden_channels[0],
                               kernel_size=3,
                               padding=1)
        self.relu1 = deltacnn.DCActivation(activation="relu")
        self.max_pool1 = deltacnn.DCMaxPooling(2)
        self.conv2 = deltacnn.DCConv2d(hidden_channels[0], hidden_channels[1],
                               kernel_size=5,
                               padding=2)
        self.relu2 = deltacnn.DCActivation(activation="relu")
        self.max_pool2 = deltacnn.DCMaxPooling(2)
        self.fc = nn.Linear(7*7*hidden_channels[1], out_features)
        self.densify = deltacnn.DCDensify()


    def forward(self, x):
        x = self.sparsify(x)
        # First convolutional layer
        x = self.conv1(x)
        # Activation function
        x = self.relu1(x)
        # Max pool
        x = self.max_pool1(x)
        # Second convolutional layer
        x = self.conv2(x)
        # Activation function
        x = self.relu2(x)
        # Max pool
        x = self.max_pool2(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layer
        x = self.fc(x)
        return self.densify(x)

# Define the channel sizes and output dim
in_channels = 3
hidden_channels = [5, 6]
out_features = 2

device = "cuda"
net = Net(in_channels, hidden_channels, out_features).to(device, memory_format=torch.channels_last)
#summary(net, (3, 28, 28), device='cuda') # (in_channels, height, width)

in_channels = 1 # Black-white images in MNIST digits
hidden_channels = [5, 6]
out_features = 10 

# Training parameters
learning_rate = 0.001
epochs = 3 

# Initialize network
net = Net(in_channels, hidden_channels, out_features)
optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

# Define list to store losses and performances of each iteration
train_losses = []
train_accs = []
test_accs = []

# Try using gpu instead of cpu
device = try_gpu()

for epoch in range(epochs):

    # Network in training mode and to device
    net.train()
    net.to(device)

    # Training loop
    for i, (x_batch, y_batch) in enumerate(train_loader):

        # Set to same device
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Set the gradients to zero
        optimizer.zero_grad()

        # Perform forward pass
        y_pred = net(x_batch)

        # Compute the loss
        loss = criterion(y_pred, y_batch)
        train_losses.append(loss)
        
        # Backward computation and update
        loss.backward()
        optimizer.step()

    # Compute train and test error
    train_acc = 100*evaluate_accuracy(train_loader, net.to('cpu'))
    test_acc = 100*evaluate_accuracy(test_loader, net.to('cpu'))
    
    # Development of performance
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    # Print performance
    print('Epoch: {:.0f}'.format(epoch+1))
    print('Accuracy of train set: {:.00f}%'.format(train_acc))
    print('Accuracy of test set: {:.00f}%'.format(test_acc))
    print('')

# Plot training curves
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(train_losses)
plt.grid()

plt.subplot(1,2,2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.plot(train_accs, label = 'train')
plt.plot(test_accs, label = 'test')
plt.legend()
plt.grid()