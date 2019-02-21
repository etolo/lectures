# Assignment 1

## P1. MNIST classification using multi-class logistic regression 

Consider a L2-regularized multi-class logistic regression model using the MNIST dataset. 

The model is given by

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{y}=\sigma(W^TX+\mathbf{b})" title="x=b" />, where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\sigma(\cdot)" title="\Large cross-entropy" /> is 
the softmax function 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;\sigma_j(\mathbf{z})=\frac{e^{z_j}}{\sum_{k=1}^Ke^{z_k}}" title="softmax" /> for <img src="https://latex.codecogs.com/svg.latex?\Large&space;j=1,2,\ldots,K" title="sum" />. 

The objective is the cross-entropy loss function:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\ell(\hat{y},y)=-y\log(\hat{y})-(1-y)\log(1-\hat{y})" title="\Large cross-entropy" />

with a L2 regularizer on the weight parameters <img src="https://latex.codecogs.com/svg.latex?\Large&space;W" title="W" />, i.e., <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda||W||^2" title="\Large cross-entropy" /> where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda" title="\Large cross-entropy" /> is a hyper-parameter. 

The hyper-parameter settings are given as below:
- minibatch size = 128 
- starting learning rate <img src="https://latex.codecogs.com/svg.latex?\Large&space;\eta^{(0)}=0.001" title="\Large cross-entropy"/>
- decaying learning rate <img src="https://latex.codecogs.com/svg.latex?\Large&space;\eta^{(t)}=\eta^{(0)}/\sqrt{t}" title="\Large cross-entropy"/> during training where <img src="https://latex.codecogs.com/svg.latex?\Large&space;t" title="t"/> is the number of epochs 
- Momentum = 0.7
- <img src="https://latex.codecogs.com/svg.latex?\Large&space;\lambda=0.01" title="\Large cross-entropy"/>
- total number of epoches = 45

**Task:** evaluate and plot **the average loss per epoch** versus the number of epoches for the training dataset, for the following optimization algorithms:
- Mini-batch gradient descent
- Mini-batch AdaGrad
- Mini-batch gradient descent with Nesterov’s momentum
- Mini-batch Adam 

Discuss how the performances of different optimization algorithms compare to each other.

## P2. CIFAR10 CNN: convergence of minibatch gradient descent

Implement a CNN architecture that consists of 3 convolutional layers followed by a hidden fully connected layer of 1000 units. 

Each convolutional layer consists of a sublayer of 5x5 convolutional filters with stride 1 followed by a sublayer of 2x2 max-pool units with stride 2. Each neuron applies ReLU activation function.

**Task:** answer the same questions as in Problem P1. In addition, show the results by adding dropout. Comment the results. 

**Hints:**

- Load CIFAR10 data by the following code:
```
from keras.datasets import cifar10
(data_train, label_train), (data_test, label_test) = cifar10.load_data()
```
- In order to reduce the training time, use only the first 50 mini-batches for each epoch. 
- More specifically, at the beginning of each epoch, randomly shuffle the whole dataset training dataset. Then, only iterate through the first 50 mini-batches for one epoch training.  
- Training on Google Colab GPU is highly recommended. The training time on 1 GPU is roughly 1 minute per epoch.  

The hyper-parameter settings:
- minibatch size = 128 
- learning rate = 0.001
- total number of epoches = 100

## P3. CIFAR10 image classification

Design and implement a convolutional neural network for the CIFAR10 image classification task aiming to achieve a high test accuracy. Evaluate the classification accuracy by reporting top-1 and top-5 test error rates. 

**Task:** plot the average loss, top-1 error rate and top-5 error rate per epoch versus the number of epochs for the training and the test dataset. 
Make sure to well describe and justify your network architecture design choices. 


**Marking scheme**:

| **Problem breakdown** | **Max marks** | 
|-------------------|---------------|
| P1 correctness of implementation	|	15 |	
| P1 discussion of results	|	15 |
| P2 correctness of implementation    |	15  |
| P2 discussion of results	|	15 |
| P3 network architecture design    |	10 |
| P3 use of different methods   | 10 |
| P3 achieved test error rates    |   10 |
| P3 discussion and presentation of results | 10 |
| | Total 100 |

