# Seminar 2: Feedforward Neural Networks
## XOR problem

### Visualise the problem
Firstly, import all the libraries we will use and fix seed for replicability, 
```
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
```
Create data, 
```
x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = np.array([[0.0], [1.0], [1.0], [0.0]])
```
and write a plotting function to visualise the original data
```
def plot_xor(x):
  plt.plot([x[0, 0], x[3, 0]], [x[0, 1], x[3, 1]], 'rs')
  plt.plot([x[1, 0], x[2, 0]], [x[1, 1], x[2, 1]], 'bs')
  plt.show()
```
Passing our data to the plotting function, as we can see in the image below, we cannot find a linear classifier that separates the red dots from the blue dots.

![alt text](https://github.com/lse-st449/lectures/raw/master/Week02/Class/graphs/xor_problem.png)

### One solution
In the lecture, we proposed this solution

![](https://github.com/lse-st449/lectures/raw/master/Week02/Class/graphs/xor_solved.png)

to find the optimal weights that solve the XOR problem. We are essentially searching for a mapping <img src="https://latex.codecogs.com/svg.latex?\Large&space;\phi" title="D_w"/> that transforms the original data into some new data that's linearly separable in the feature space.

We will therefore create a two-layer neural network to solve the XOR problem.

### Intuition for variable sharing in a neural network
Consider the linear regression example in Semiar 1.  We are fitting the model <img src="https://latex.codecogs.com/svg.latex?\Large&space;y=wx+b" title="D_w"/> to all the data points, i.e., the parameters are shared across the entire dataset.

Similarly, the same logic applies when using neural networks.  Assume that we have built the following network for the XOR problem, which has one hidden layer with four neurons and an output layer with one neurons, 

<img src="https://github.com/lse-st449/lectures/raw/master/Week02/Class/graphs/NN.png" width="700" height="450">

Each forward pass takes only one data point which has a dimensionality of 2. To find the optimal parameters for the entire dataset, we need to make sure that the same set of parameters are updated in the training process. 

## Variable sharing in TensorFlow
### Variable scope
Variable scope is a mechanism in Tensorflow that allows users to share variables accessed in different parts of the code. 
```
tf.variable_scope(scope, reuse=None, initializer=None)
```
`tf.variable_scope` also allows you to reuse the code for creating variables.  For example, instead of writing
```
w1 = tf.get_variable("weights", [2, 2], dtype=tf.float64, initializer=tf.random_normal_initializer(stddev=0.1))
b1 = tf.get_variable("bias", shape=[1, 2], dtype=tf.float64, initializer=tf.constant_initializer(0.0))
w2 = tf.get_variable("w2", shape=[2, 1], dtype=tf.float64, initializer=tf.random_normal_initializer(stddev=0.1))
b2 = tf.get_variable("bias2", shape=[1], dtype=tf.float64, initializer=tf.constant_initializer(0.0))
```
we can write
```
with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
    w = tf.get_variable("weights", [2, 2], dtype=tf.float64, initializer=tf.random_normal_initializer(stddev=0.1))
    b = tf.get_variable("bias", shape=[1, 2], dtype=tf.float64, initializer=tf.constant_initializer(0.0))
```
`tf.AUTO_REUSE` will fetch variables if they already exist and create new ones if they don't.  Assume that we name our variable scopes in the first hidden layer as `h1` and in the second layer as `h2`.  If the argument `scope='h1'`, it create `w1` and `b1`.  You can continue to create `w2` and `b2` by using a different scope `scope='h2'`.  Now, if you want to reuse the `w1` and `b1` that already exist,  set the argument `scope = 'h1'` again.  

### Implement the neural network

#### Compute the dimensionality of parameters
<img src="https://github.com/lse-st449/lectures/raw/master/Week02/Class/graphs/NN.png" width="700" height="450">

We see that the dimensionality of the weight matrix `D_w` and of the bias `D_b` are  <img src="https://latex.codecogs.com/svg.latex?\Large&space;D_w=[N^{(l)},N^{(l-1)}]" title="D_w"/>  and  <img src="https://latex.codecogs.com/svg.latex?\Large&space;D_b=[N^{(l)},1]" title="D_b"/>, where `l` is the current layer and `l-1` is the previous layer. 

Given the dimensionalities of the weights and bias, you are now able to compute the total number of parameters of your neural network. In this example, we have 17 parameters. 

Let's create a neural network to solve the XOR problem using what we have learned so far.  
#### Create a layer in neural network
```
def fully_connected_nn(x, output_dim, scope):  
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):  
        w = tf.get_variable("weights", shape=[output_dim, x.get_shape()[0]], dtype=tf.float64,  
        initializer=tf.random_normal_initializer(seed=100, stddev=0.1))  
        b = tf.get_variable("bias", shape=[output_dim, 1], dtype=tf.float64, 
        initializer=tf.constant_initializer(0.0))  
    return tf.add(tf.matmul(w, x), b)
```
Note that the initialisation for parameters is essential. 

#### Define a two-layer neural network
```
def two_layer_nn(x):
    hidden_layer1 = tf.nn.relu(fully_connected_nn(x, 2, 'h1'))
    output = tf.nn.sigmoid(fully_connected_nn(hidden_layer1, 1, 'h2'))
    return output
```

#### Create the Graph and Specify the Loss Function
Cross-entropy loss is typically used to measure the performance of a classifier that outputs a value between 0 and 1. 
The cross-entropy loss is given by

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{L}=-y\log(\hat{y})-(1-y)\log(1-\hat{y})" title="\Large cross-entropy" />

We implement the loss function for later use:
```
def cross_entropy(y, y_hat):
    loss = 0.0
    m = y.shape[0]
    for i in range(m):
        if y[i] == 1:
            loss += -tf.log(y_hat[i])
        else:
            loss += -tf.log(1.0 - y_hat[i])
    n = tf.cast(m, tf.float64)
    return loss / n
```
Build the graph, 
```
n_data = x.shape[0]  
y_hat = []  
  
data_x = tf.placeholder(tf.float32, shape=[None, 2])  
data_y = tf.placeholder(tf.float32, shape=[None, 1])  
for i in range(n_data):  
    h1, output = two_layer_nn(tf.reshape(x[i, :], [2, 1]))  
    y_hat.append(tf.squeeze(output)) # convert the [1, 1] tensor output to a scalar  
    hidden.append(h1)  
loss = cross_entropy(y, y_hat)  
optimiser = tf.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)
```
Here we have built a single layer neural network and `h` is the transformed data that is output by the hidden layer. 

Execute the graph with a session:
```
n_iterations = 3000  
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
    loss_list = []  
    for i in range(n_iterations):  
        _, l = sess.run([optimiser, loss], feed_dict={data_x: x, data_y: y})  
        loss_list.append(l)
```
Plot the loss by 
```
s = np.arange(n_iterations)
plt.plot(s, loss_list)
plt.title('Loss')
plt.show
```

![alt text](https://github.com/lse-st449/lectures/raw/master/Week02/Class/graphs/loss_xor.png)

We also fetch the learned`y_hat` values after the training, 
```
y_tr = sess.run(y_hat)
print('Prediction:', np.round(y_tr))
```
Output:
```
Prediction: [0. 1. 1. 0.]
```
and the we can also fetch `h1` values to plot a latent representation of the original inputs:

![alt text](https://github.com/lse-st449/lectures/raw/master/Week02/Class/graphs/xor_trained.png)

As shown and explained in Figure 6.1 (Chapter 6, p.p 168) in Deep Learning (Goodfellow, Bengio, and Courville), the two points that must have the same output have been collapsed in a single point in feature space. 

## Build a Neural Network to Solve the XOR problem using Tensorflow
`tf.layers.dense` allows you to add a fully connected layer to your network. 
```
tf.layers.dense(inputs, units, activation=None,use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, name=None, reuse=None)
```
We now rewrite our solution using Tensorflow. Firstly, define the neural network, 
```
def xor_network(x):  
    with tf.variable_scope('xor', reuse=tf.AUTO_REUSE) as scope:  
        h = tf.layers.dense(x, 2, activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.random_normal_initializer(seed=100, stddev=0.1))  
        y = tf.layers.dense(h, 1, activation=None, use_bias=True, kernel_initializer=tf.random_normal_initializer(seed=100, stddev=0.1))  
    return y
```
Then build your graph, 
```
data_x = tf.placeholder(tf.float64, shape=[None, 2])  
data_y = tf.placeholder(tf.float64, shape=[None, 1])  
  
h, y_hat = xor_network(data_x)  
y_predict = tf.nn.sigmoid(y_hat)  
  
vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='xor')  
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_y, logits=y_hat))  
optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss, var_list=vars)
```
`vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='xor')` demonstrates another usage of variable scope.  One can get the collection of variables in a specific scope and minimise the loss function only w.r.t variables in this scope, i.e., `.minimize(loss, var_list=vars)`.  

Next, execute your graph, 
```
n_iterations = 3000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = []
    for i in range(n_iterations):
        _, l = sess.run([optimiser, loss], feed_dict={data_x: x, data_y: y})
        loss_list.append(l)
```
To fetch the learned `y`, 
```
y_pred = sess.run(y_predict, feed_dict={data_x: x})
print('Prediction:', np.round(y_pred))
```
Output:
```
Prediction: [[0.] [1.] [1.] [0.]]
```
## Visualise Your Neural Networks
You can use the library called **ANNvisualizer** to visualise your neural networks.  Unfortunately this is not yet compatible with any Python notebooks.  If you have installed a Python IDE, you can follow the instructions to install the library.  
For Windows users, open your Anaconda Prompt and activate your virtual environment, and then install `graphviz`, `keras` and `ann_visualizer` by
```
conda install -c anaconda graphviz
```
```
pip3 install keras
```
```
pip3 install ann_visualizer
```
And for Linux/macOS, you can use the following commands after activating your virtual environment,
```
sudo apt-get install graphviz && pip3 install graphviz
```
```
pip3 install keras
```
```
pip3 install ann_visualizer
```
To create the neural network we just implemented, we simply add two layers to the graph. The first layer has an input dimensionality of 2 and 2 neurons(units), and the output layer has 1 neuron. 
```
from keras.models import Sequential  
from keras.layers import Dense  
from ann_visualizer.visualize import ann_viz  

network = Sequential()  
#Hidden Layer#1  
network.add(Dense(units=2, activation='relu',  kernel_initializer='uniform', input_dim=2))  
#Output Layer  
network.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))  
ann_viz(network, title="")
```

![alt text](https://github.com/lse-st449/lectures/raw/master/Week02/Class/graphs/2-layer_nn.png)

An example with more layers and neurons,
```
network = Sequential()  
#Hidden Layer#1  
network.add(Dense(units=10, activation='relu', kernel_initializer='uniform', input_dim=5))  
#Hidden Layer#2
network.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))  
#Hidden Layer#3
network.add(Dense(units=3, activation='relu', kernel_initializer='uniform'))  
#Output Layer  
network.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))  
ann_viz(network, title="")
```

![alt text](https://github.com/lse-st449/lectures/raw/master/Week02/Class/graphs/multi-layer_nn.png)

This library can also visualise convolutional neural networks, which can be useful for our future courses. 

## Homework: Single-layer perceptron algorithm
### Generate Input
- Fix random seed using the following code:
```
np.random.seed(1)
```
- Sample 200 data points from an uniform distribution at random on the rectangle [<img src="https://latex.codecogs.com/svg.latex?\Large&space;\gamma" title="D_w"/> /2, 1] x [-1/2, 1/2] and another 200 at random on the rectangle [-1, -<img src="https://latex.codecogs.com/svg.latex?\Large&space;\gamma" title="D_w"/>/2] x [-1/2, 1/2], where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\gamma" title="D_w"/> is a margin parameter that decides how closes these two classes are. See figure below for an illustration:

![](https://github.com/lse-st449/lectures/raw/master/Week02/Class/graphs/percep_data.png)

```
gamma = 0.1  
a1 = np.random.uniform(-1.0, -gamma / 2.0, size=200)  
a2 = np.random.uniform(-0.5, 0.5, size=200)  
  
x1 = np.random.uniform(gamma/2.0, 1.0, size=200)  
x2 = np.random.uniform(-0.5, 0.5, size=200)  
  
c1 = np.squeeze(np.dstack((a1, a2)))  
c2 = np.squeeze(np.dstack((x1, x2)))
```
- Generate labels, concatenate samples in both classes and randomly shuffle them:
```
# create labels  
y1 = np.negative(np.ones(200))  
y2 = np.ones(200)  
  
data = np.concatenate((c1, c2))  
labels = np.concatenate((y1, y2))  
  
# shuffle the data in the same way  
randomize = np.arange(len(data))  
np.random.shuffle(randomize)  
data = data[randomize]  
labels = labels[randomize]
```
### Tasks
- Plot the data in the two classes
- Implement the perceptron learning algorithm (**slide 11-15**) in TensorFlow (with <img src="https://latex.codecogs.com/svg.latex?\Large&space;\phi" title="D_w"/> = identity mappingm, i.e., <img src="https://latex.codecogs.com/svg.latex?\Large&space;\phi(x)=x" title="D_w"/>)
- Run the algorithm for the value of parameter <img src="https://latex.codecogs.com/svg.latex?\Large&space;\gamma=0.1" title="D_w"/>
- Plot the decision boundary after training 
- Record the total number of mistakes made by the algorithm

**Hints**
- This is online learning, meaning that you do not pass the whole dataset before making a parameter update. The loss and parameter update are computed after each data point is passed to the computational graph. 
- The parameter update rule is stated on **slide 15**. When assigning values to variables in TensorFlow, use 
```
tf.assign(ref, value, validate_shape=None, use_locking=None, name=None)
```
- Stop as soon as there are no misclassified samples
- See figure below for an illustration of decision boundary:

![](https://github.com/lse-st449/lectures/raw/master/Week02/Class/graphs/decision_boundary.png)
