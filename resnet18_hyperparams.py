import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def plot_kernel(model):
    model_weights = model.state_dict()
    fig = plt.figure()
    plt.figure(figsize=(10,10))
    for idx, filt  in enumerate(model_weights['conv1.weight']):
        if idx >= 32: continue
        plt.subplot(4,8, idx + 1)
        plt.imshow(filt[0, :, :], cmap="gray")
        plt.axis('off')
    
    plt.show()

def plot_kernel_output(model,images):
    fig1 = plt.figure()
    plt.figure(figsize=(1,1))
    
    img_normalized = (images[0] - images[0].min()) / (images[0].max() - images[0].min())
    plt.imshow(img_normalized.numpy().transpose(1,2,0))
    plt.show()
    output = model.conv1(images)
    layer_1 = output[0, :, :, :]
    layer_1 = layer_1.data

    fig = plt.figure()
    plt.figure(figsize=(10,10))
    for idx, filt  in enumerate(layer_1):
        if idx >= 32: continue
        plt.subplot(4,8, idx + 1)
        plt.imshow(filt, cmap="gray")
        plt.axis('off')
    plt.show()

def test_accuracy(net, dataloader):
  ########TESTING PHASE###########
  
    #check accuracy on whole test set
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test set: %d %%' % (
    accuracy))
    return accuracy
 

####RUNNING CODE FROM HERE:
      
#transform are heavily used to do simple and complex transformation and data augmentation
transform_train = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

transform_test = transforms.Compose(
    [
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=4,drop_last=True)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=4,drop_last=True)


dataiter = iter(trainloader)

###Show images:
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
###
n_classes = 100 
####
#Residual Network:
net = models.resnet18(pretrained=True)
net.fc = nn.Linear(512, n_classes) 
####

net = net.cuda()

criterion = nn.CrossEntropyLoss().cuda()
# Learning Rate: 1e-3
optimizer = optim.Adam(net.parameters(), lr=1e-3) 
# Learning Rate: 1e-5
# optimizer = optim.Adam(net.parameters(), lr=1e-5) 
# Learning Rate: 0.1
# optimizer = optim.Adam(net.parameters(), lr=0.1) 
# Learning Rate: 1e-3, weight_decay = 0.1
# optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay = 0.1) 
# Learning Rate: 1e-3, weight_decay = 1e-5
# optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay = 1e-5) 
# Learning Rate: 1e-5, weight_decay = 1
# optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay = 1) 


########TRAINING PHASE###########
n_loss_print = len(trainloader)  #print every epoch, use smaller numbers if you wanna print loss more often!

losses=[]
accuracy = []

n_epochs = 10
for epoch in range(n_epochs): 
    net.train() 
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs and cast them into cuda wrapper
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % n_loss_print == (n_loss_print -1):    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / n_loss_print))
            losses.append(running_loss / n_loss_print)
            running_loss = 0.0
    accuracy.append(test_accuracy(net,testloader))

print('Finished Training')
    
plt.title('Training loss & accuracy curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.plot(range(n_epochs),accuracy, label='Accuracy')
plt.plot(range(n_epochs),losses, label='Loss')
plt.legend()