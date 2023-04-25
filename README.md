# DeltaCNN_Reproduction
This repository contains the code and the blogpost that document our efforts towards trying to replicate the results of the paper ***"DelatCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos"***. The sutdy was conducted to satisfy the requirements for the course ***Deep Learning (CS4240)*** at TU Delft. Below we briefly mention about the contents of the files and folder in the repository:
1. simple_cnn.py: Contains the code where we used to employ the DeltaCNN backend in order to replace the Pytorch CNN layers with the sparse DeltaCNN layers
2. resnet18_hyperparams.py: File containing the hyperparameter tuning performed on the ResNet-18 architecture using the CIFAR-100 dataset
3. DeltaCNN Folder: Contains the original code of the study we tried to replicate
4. blogpost.md: The written blogpost detailing the issues faced and the results obtained during the course of this project 
