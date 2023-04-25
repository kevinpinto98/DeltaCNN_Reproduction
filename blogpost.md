# DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos

***Group***: 48
***Members***:
* Kevin Pinto (5709202): ResNet-18 hyperparameter tuning, DeltaCNN implementation of basic CNN, attempted to download Human3.6M dataset and writing of the blogpost
* Julia Dijkstra (4607848): Attempted to download Human3.6M dataset, get DeltaCNN working on pose-resnet and a simplified CNN, helped with writing the blogpost
* Andrei Popovici (5853060): Setting up the pose-resnet architecture, tried to download the Human3.6M dataset and tried to get the DeltaCNN framework working

This project was done in part to satisfy the requirements for the course ***Deep Learning (CS4240)*** at TU Delft.
We were tasked with reproducing the results of the paper ***"DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos"***. In this blog we'll be elaborating on our efforts to reproduce the results, the issues we faced and the alternate experiments we conducted with the ResNet-18 architecture and the CIFAR-100 dataset. 

## Introduction
DeltaCNN is a framework that aims to provide sparse implementation for all CNN layers and enable sparse frame-by-frame updates to accelerate video inference with minimal loss in accuracy.

It mainly exploits the fact that a large number of video frames typically change very little and hence by skipping identical image regions we can reduce the computational resources utilized to perform CNN inference on video data. The way it works is that intermediate feature maps from previous frames are cached in order to accelerate the inference of new frames by processing the updated pixels.

The sparse implementation of CNN layers provided by DeltaCNN can be utilized to replace the PyTorch layers with the DeltaCNN equivalent layers. An important point to note is that DeltaCNN only works on GPUs with CUDA support since all the layers are implemented in CUDA.

## Project Goals
The aim of this project was to reproduce the results for one of the architectures presented in Table 1 of the paper as shown below:![](https://i.imgur.com/TceFxXB.png)

Due to the unavailability of all the devices mentioned in the table we initially aimed to demonstrate the speedup that can be achieved using the DeltaCNN backend on a simple CNN architecture compared to its PyTorch implementation. However, due to the issues we faced that we describe later, we consulted with our TA and instead focused on understanding the behaviour of the ResNet-18 architecture using the CIFAR-100 dataset and observing the results obtained after performing hyperparameter tuning.

## Methodology
The first step for the primary goal of using DeltaCNN is to install the relevant packages in Google Colab which was our environment of choice for working on this project.
Installing the DeltaCNN packages can be done using the following commands in Colab:
```python=
!git clone https://github.com/facebookresearch/DeltaCNN.git
!pip install /content/DeltaCNN/
```
The next step involves replacing the relevant PyTorch layers with the DeltaCNN equivalent. An example of this was demonstrated in the DeltaCNN GitHub repository and is shown replicated below for convenience:
```python
####### PyTorch
from torch import nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(...)
        self.conv2 = nn.Conv2d(...)
        self.conv3 = nn.Conv2d(...)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.relu(self.conv3(x))
```

```python
####### DeltaCNN
import deltacnn
class CNN(deltacnn.DCModule):
    def __init__(self):
        super(CNN, self).__init__()
        self.sparsify = deltacnn.DCSparsify()
        self.conv1 = deltacnn.DCConv2d(...)
        self.conv2 = deltacnn.DCConv2d(...)
        self.conv3 = deltacnn.DCConv2d(...)
        self.relu1 = deltacnn.DCActivation(activation="relu")
        self.relu2 = deltacnn.DCActivation(activation="relu")
        self.relu3 = deltacnn.DCActivation(activation="relu")
        self.densify = deltacnn.DCDensify()

    def forward(self, x):
        x = self.sparsify(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.densify(self.relu3(self.conv3(x)))
```
Although we performed the above mentioned steps we continued to face several errors and despite our best efforts we were unable to resolve them. In addition to the errors due to the DeltaCNN backend we also faced issues in obtaining the Human3.6M dataset. The issues faced will be explained in detail in the following section.
### Issues Faced
1. Inability to obtain the Human 3.6M dataset
* The Human3.6M dataset consists of 3.6 million 3D human poses by both male and female actors in several different situations such as standing, talking on the phone, smoking etc.
* In order to obtain the dataset we needed to contact the reponsible entity however even at the time of this writing we have not yet received a response.
* Our project supervisor then suggested to us to use the following link: [https://blog.csdn.net/qq_42951560/article/details/126380971](https://) 
Using this we managed to download the dataset however we soon found that there were several missing annotations in the dataset which overall made it unsuitable to be used for the study.
* Our conclusion for the dataset remains that unless one can obtain it from the original source i.e. [http://vision.imar.ro/human3.6m/description.php](https://) it is not possible to get the dataset.

2. Errors when trying to replace the PyTroch layers with the DeltaCNN sparse layer implementation
* When we tried to replace the PyTroch layers with the DeltaCNN equivalent we obtained the following errors:
(2.1)
![](https://i.imgur.com/lBHL32F.png)
The error mentioned here could be resolved by changing the device on which the code is run to a GPU.
(2.2)
![](https://i.imgur.com/hEaHi7Q.png)
(2.3)
![](https://i.imgur.com/3TXChiV.png)
The errors mentioned in 2.2 and 2.3 could not be resolved and we kept on encountering them despite our repeated attempts to resolve the said errors.
The file "simple_cnn.py" contains the code we used to employ the DeltaCNN backend in order to replace the Pytorch CNN layers with the sparse DeltaCNN layers.

## ResNet-18 Study
Since we were unable to run the DeltaCNN framework we consulted with our TA and decided to perform some experiments using the ResNet-18 CNN architecture in combination with the CIFAR-100 dataset and perform hyperparamter tuning to observe the results on training loss and accuracy.
The code for our hyperparameter tuning experiments with ResNet-18 can be found in the file "resnet18_hyperparams.py".

### Hyperparameter Tuning
Hyperparamter tuning refers to the process of choosing the optimal set of hyperparameters for a learning algorithm. In our study we mainly focused on tuning the hyperparameters for the Adam optimizer. The various configurations of the hyperparameters used are as follows:
* Learning Rate: 1e-3
* Learning Rate: 1e-5
* Learning Rate: 0.1
* Learning Rate: 1e-3, weight_decay = 0.1
* Learning Rate: 1e-3, weight_decay = 1e-5
* Learning Rate: 1e-5, weight_decay = 1

## Results
Mentioned below are the results of our experiments with hyperparameter tuning i.e. the network accuracy  and the corresponding plots of the accuracy/loss curves vs the number of epochs. 
1. Learning Rate: 1e-3
![](https://i.imgur.com/Md1DjfH.png)
![](https://i.imgur.com/78BZzpn.png)

2. Learning Rate: 1e-5
![](https://i.imgur.com/hstOQy0.png)
![](https://i.imgur.com/6ZUy3tb.png)

3. Learning Rate: 0.1
![](https://i.imgur.com/wQiPQ0C.png)
![](https://i.imgur.com/QoVs3uK.png)

4. Learning Rate: 1e-3, weight_decay = 0.1
![](https://i.imgur.com/eDUsH92.png)
![](https://i.imgur.com/LErC4by.png)

5. Learning Rate: 1e-3, weight_decay = 1e-5
![](https://i.imgur.com/UnYUZSY.png)
![](https://i.imgur.com/H7wGtuL.png)

6. Learning Rate: 1e-5, weight_decay = 1
![](https://i.imgur.com/kx7gcX5.png)
![](https://i.imgur.com/MlXc080.png)

## Discussion
From the figures of accuracy/loss in the previous section we can observe that using a large learning rate results in a very low accuracy rate since using a larger step in the learning algorithm might result in being stuck at a point far from the minima of the loss function. Similarly, using a very low value of the learning rate causes the model to learn very slowly and it may not have reached the minima in the required number of epochs thus resulting in a comparatively lower accuracy than when the larning rate was 1e-3.

In addition to the learning rate we have also observed the effect of weight decay on the values of accuracy/loss. Since weight decay represents the L2 penalty which is a regularization technique that avoids the risk of overfitting and shrinks the coefficient estimates towards zero we notice that high values of weight decay result in lower accuracy. This is mainly due to the fact that even though weigh decay avoids overfitting a high value of it also prevents the model from learning in a reasonable manner.

## Conclusion
Based on our discussion in the previous sections with regard to DeltaCNN we conclude that it is incredibly difficult to replicate the results of it's study due to the issues already mentioned. Due to time contraints, a lack of the necessary computational resources, inability to obtain the dataset and the ambiguous nature of the errors that made them hard to debug we were unable to recreate the results for DeltaCNN.

We instead focused on performing the hyperparameter tuning for the ResNet-18 architecture trained on the CIFAR-100 dataset. The results for hyper-parameter tuning obtained agree with our theoretical predictions for accuracy/loss values vs the number of epochs for several different cases of the "learning_rate" and the "weight_decay".

## References
1. M. Parger, et al., "DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos," in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA, 2022 pp. 12487-12496.
2. DeltaCNN Github Repository: [https://github.com/facebookresearch/DeltaCNN](https://)
3. ResNet-18 PyTorch Documentation: [https://pytorch.org/vision/master/models/generated/torchvision.models.resnet18.html](https://)
4. Human3.6M Dataset: [http://vision.imar.ro/human3.6m/description.php](https://)