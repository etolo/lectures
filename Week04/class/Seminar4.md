# Seminar 4: Convolutional Neural Networks (CNN)

The goal of this seminar is to learn how to create and train a convolutional neural network. We consider the task of handwritten digit recognition.

## Image data

### B&W images
As an example for B&W images (obtained from MNIST dataset):
![alt text](https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/mnist3.png)
This image has a resolution of 28x28 and 1 channel. Each pixel is between a number between 0 and 1, where 0 is shown as black, 1 is white and any number inbetween is gray.  The image data is presented as a flattened long vector with a length of 784(28x28).  

### Colour images
![alt text](https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/cat.png)
In this beautiful cat photo, one pixel has 3 dimensions, i.e., the RGB channels.  The shape of this image is 178x218x3.  
Each element in one pixel is a number between 0 and 255.  

How are colour images stored?  
**Different datasets store data in different shapes.**
Use CIFAR-10 dataset as an example (each image has a shape of 32x32x3):

If you download the data from some websites, it may stored as a numpy array of uint8s with a length of 3072(32x32x3). The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.

However, if you load the dataset from `tf.keras`, the data is stored of shape 32x32x3. For 10000 training images, the training set has a shape of 10000x32x32x3. 

## MNIST dataset

MNIST dataset contains 55,000 training data points and 10,000 test data points with corresponding labels. Each training data point in the MNIST dataset is a 28x28 pixel grey-scale image of a handwritten digit (0-9).  

### Import the dataset

TensorFlow has the dataset already built in, so there is no need to manually download it.

```
from tensorflow.examples.tutorials.mnist import input_data

# read the data 
mnist=input_data.read_data_sets("mnist", one_hot=True)

# import training and test sets and the corresponding labels
training_set = mnist.train.images
training_labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_set = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# Shapes of training set
print("Training set images shape: {shape}".format(shape=mnist.train.images.shape))
print("Training set labels shape: {shape}".format(shape=mnist.train.labels.shape))

# Shapes of test set
print("Test set images shape: {shape}".format(shape=mnist.test.images.shape))
print("Test set labels shape: {shape}".format(shape=mnist.test.labels.shape))
```
Output:

![alt text](https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/data_shape.png)

What `one_hot=True` does is to represent the label "8" by using a vector of 10 elements, all equal to zero but the 8th element equal to 1. As shown in the outputs, the images are actually represented as a long vector with dimensionality of 784 (28x28).

**Tips**:  If you see warnings when importing MNIST, add `tf.logging.set_verbosity(tf.logging.INFO)` in the header. 

### Visualize the data

If we want to visualize 8 by 8 images randomly chosen from the training set, we pass arguments `training_set`, `row=8`, and `col=8` to the function below, 

```
def display(x, row, col):
    num_of_images = row * col
    fig, axe = plt.subplots(row, col, figsize=(8, 8))

    for i in range(row):
        for j in range(col):
            axe[i][j].imshow(x[np.random.randint(0, num_of_images), :].reshape(28, 28), origin="upper", cmap="gray",
                             interpolation="nearest")
            axe[i][j].set_xticks([])
            axe[i][j].set_yticks([])
    plt.show()
```
![alt text](https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/mnist_plot.png)

## MNIST classification using CNN

###  Basic architechture
<img src="https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/mnist_cnn.png" width="900" height="450">

### Build the model
#### Reshape the inputs

The 2D convolutional layer `tf.layers.conv2d` takes inputs of shape `[batch_size, height, width, channels]`. We therefore convert the original data shape `[1, 784]` using `tf.reshape` operation:
```
inputs = tf.reshape(data, [-1, 28, 28, 1])
```
`-1` is used to infer the first dimension.  For example,  if `data` contains 7840 values, `inputs` will have a shape of `[10, 28, 28, 1]`.

#### Convolutional layer
```
tf.layers.conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, ...)
```
**Example:**
```
conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[5, 5], strides=(1, 1), padding="same", activation=tf.nn.relu)
```
`filter` is the number of filters to use, which is also dimensionality of outputs. `kernel_size` specifies the `[height, width]` of the filter.  

The `padding` argument takes one of the values: `valid` or `same`. `same` preserves the shape of input tensor by adding 0 values to the edges and `valid` may drop the columns on the right or rows at the bottom. 

The `stride` argument is set to be `(1, 1)` by default. 

In this example, the output tensor produced by this layer has a shape of `[batch_size, 28, 28, 32]`: the same height and width dimensions as the input, but now with 32 channels holding the output from each of the filters.

#### Max pooling layer
```
tf.layers.max_pooling2d(inputs, pool_size, strides, padding='valid', data_format='channels_last', name=None)
```
**Example:**
```
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
```

Inputs of `tf.layers.max_pooling2d` also needs to have rank 4.  The `pool_size` argument specifies `[pool_height, pool_width]` and the `strides` argument defines the strides of the pooling operation. By choosing a stride of 2 and pooling size of 2x2, it means there will be no overlap in the input patches used. 

The output shape of this layer is `[batch_size, 14, 14, 32]`.

#### Dropout layer
```
dropout1 = tf.layers.dropout(inputs=pool1, rate=0.2, training=training)
```
The `rate` argument takes a value between 0 and 1. `rate=0.2` would drop out 20% of the input units. 

The `training` argument takes a boolean. `training=True` will apply dropout while `training=False` returns the input untouched. 

#### Dense layer

Our CNN returns an output tensor with a shape of `[batch_size, 7, 7, 64]`. We want to add a feedforward neural network to map the features extracted by CNN to the target classes.  To connect this dense layer, we need to flatten the feature map to an input tensor of shape `[batch_size, features]`.

```
p2_flat = tf.reshape(dropout2, [-1, 7 * 7 * 64])  # 3136 dims
```
Now we add a fully connected layer, reducing the dimensionality to 1024, 
```
h1 = tf.layers.dense(inputs=p2_flat, units=1024, activation=tf.nn.relu)
```
After applying another dropout layer, we finally map the input to 10 dimensions, 
```
outputs = tf.layers.dense(inputs=dropout3, units=10, activation=None)
```

Let's put this CNN together. 

```
def classification_cnn(data, training=False):
    with tf.variable_scope('classification_cnn', reuse=tf.AUTO_REUSE):
        # reshape the inputs
        inputs = tf.reshape(data, [-1, 28, 28, 1])

        # convolutional layer 1
        conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        # pooling layer 1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        # dropout layer 1
        dropout1 = tf.layers.dropout(inputs=pool1, rate=0.2, training=training)

        # layers 2
        conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        dropout2 = tf.layers.dropout(inputs=pool2, rate=0.2, training=training)
        
        # flatten inputs for dense layer
        p2_flat = tf.reshape(dropout2, [-1, 7 * 7 * 64])  # 3136 dims

        # dense layer
        h1 = tf.layers.dense(inputs=p2_flat, units=1024, activation=tf.nn.relu)
        dropout3 = tf.layers.dropout(inputs=h1, rate=0.2, training=training)
        
        # output layer
        outputs = tf.layers.dense(inputs=dropout3, units=10, activation=None)
    return outputs
```

The reason why we didn't apply a softmax transformation is because the Tensorflow built-in cross-entropy loss function applies the softmax function internally. 

### Loss function 

#### Softmax activation for classification

The softmax function is generalization of the logistic function, defined as

<img src="https://latex.codecogs.com/svg.latex?\Large&space;\sigma(\mathbf{z})_j=\frac{e^{z_j}}{\sum_{k=1}^Ke^{z_k}}" title="\Large cross-entropy" /> , for j = 1, ..., K. 

The output of the softmax function can be seen as a probability distribution over K possible outcomes.  

#### TensorFlow softmax cross-entropy loss 

There are two different versions of softmax cross-entropy loss function available in TensorFlow.  
```
tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, dim=-1, name=None)
```
The `labels` argument takes inputs of shape `[batch_size, num_classes]`, i.e., each row should be a one-hot vector.  The `logits` argument expects unscaled logits, i.e., the output from the final layer of our model without activation functions.  
```
tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)
```
The difference in the sparse version is the `labels` argument that takes a vector of indices in [0, num_classes).

## Visualise the network with ANN visualizer

```
import keras
from ann_visualizer.visualize import ann_viz

def build_cnn_model():
  model = keras.models.Sequential()

  model.add(keras.layers.Conv2D(32, (5, 5), padding="same",input_shape=(28, 28, 1), activation="relu"))

  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
  model.add(keras.layers.Dropout(0.25))

  model.add(keras.layers.Conv2D(64, (5, 5), padding="same", input_shape=(14, 14, 1), activation="relu"))

  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
  model.add(keras.layers.Dropout(0.25))

  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(1024, activation="relu"))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(10, activation="softmax"))

  return model

model = build_cnn_model()
ann_viz(model, title="")
```
With the above code, you should be able to produce a CNN graph like [this](https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/cnn.pdf).

Use `model.summary()` to get a summary of the model,

![alt text](https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/model_summary.png)

### Train the model

Define the parameters needed
```
batch_size = 64
x_dim = 784
num_classes = 10
```
Create placeholders and define the computational graph, 
```
x = tf.placeholder(tf.float32, [batch_size, x_dim])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
y_labels = tf.argmax(y, axis=1)
outputs = classification_cnn(x, training=True)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_labels, logits=outputs))
optimiser = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
```
Since we are using `tf.nn.sparse_softmax_cross_entropy_with_logits`, we first convert the one-hot representations of labels using ` tf.argmax(y, axis=1)` which returns the index with the largest value across columns of a tensor.

### Generate predictions

Create indices for computing the accuracy,
```
test_labels_idx = tf.argmax(test_labels, axis=1)
```
```
preds = classification_cnn(test_set, training=False)
predictions = tf.nn.softmax(preds)
predictions_idx = tf.argmax(predictions, axis=1)
accuracy, update_op = tf.metrics.accuracy(labels=test_labels_idx, predictions=predictions_idx)

```
In the stage of testing,  the `training` argument of the model should be set as `False`. 

`tf.metrics.accuracy` computes the frequency with which `predictions` matches `labels`.  This methods creates two local variables: `total` and `count` and returns two outputs: `accuracy` and `update_op`.  When `predictions` matches `labels`, the internal operation `is_correct` returns 1, otherwise 0.  In execution, both `accuracy` and `update_op` need to be called, otherwise `count` won't be updated.

The part of code is added by the end of the computational graph in the previous sector as we want to record accuracy per iteration. 

### Execute the graph
```
iterations = 30000
with tf.Session() as sess:
  print('Start')
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  loss_list = []
  accuracy_list = []

  for i in range(iterations):
    batch_data, batch_labels = mnist.train.next_batch(batch_size)
    _, l, acc, acc_op = sess.run([optimiser, loss, accuracy, update_op], feed_dict={x: batch_data, y: batch_labels})
    loss_list.append(l)
    
    acc1 = sess.run(accuracy)
    accuracy_list.append(acc1)
    if i % 100 == 0 and i != 0:
      print('Iteration: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(i, l, acc1))
      plot_loss_accuracy(loss_list, accuracy_list)
```
Because there are two local variables within `tf.metrics.accuracy`, in addition to initialising the global variables, we also have to initialise local variables using
```
sess.run(tf.local_variables_initializer())
```
The built-in function
```
mnist.train.next_batch(batch_size)
```
allows you to obtain mini-batches easily. 

**Warning:** 
- In the loop, you have to fetch the value of  `accuracy` after the both `accuracy` and `update_op` are called, otherwise it's not updated.
- Train the model until convergence.  30,000 iterations may not be enough. 

## Training tricks
### Dropout
It's very common for NNs to suffer from over-fitting.  We first run the algorithm without dropout layers.  After 200 iterations, we observe that the objective loss is 0.0003 and prediction accuracy is 92.06%. 

<img src="https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/nodropout_loss.png" width="550"/> <img src="https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/nodropout_acc.png" width="550"/>  

To improve the prediction accuracy, we add dropouts by setting `training=True` in the experiment. With `dropout rate = 0.2` in the CNN layers and dense layers, the accuracy is increased to 93.56% after 200 iterations.

<img src="https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/dropout0.2loss.png" width="550"/> <img src="https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/dropout0.2acc.png" width="550"/>   

### Learning rate decay

When training a model, it is often a good practice to lower the learning rate as the training progresses. As observed from the results above, the loss of the model is close to zero during most of the training process. 
The decayed learning rate is computed by
```
decayed_learning_rate = learning_rate * decay_rate^(global_step / decay_steps)
```
Tensorflow has built such a function for users:
```
tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)
```
`learning_rate` is your starting learning rate. `global_step`is your current iteration step, used for the decay computation together with `decay_steps`. `decay_rate` is the base of the decay.   When `staircase=True`, `global_step / decay_steps` is an integer division.

To use `tf.train.exponential_decay`,  add `global_step` as a placeholder in the computational graph,
```
step = tf.placeholder(tf.int32)
```
Then define the parameters,
```
import math

starting_learning_rate = 1e-3
decay_step = 2000
decay_rate = 1/math.e
```
Compute the updated learning rate in your computational graph, 

```
new_lr = tf.train.exponential_decay(starting_learning_rate, step, decay_step, decay_rate)
```

This means that when your iteration step is 2,000, your learning rate is 0.00037. When your iteration step is 4,000, your learning rate decays to 0.00014. 

In the execution phrase, you will need to pass the "step" parameter to your model at every iteration through the  `feed_dict`  in the loop. 
```
sess.run([optimiser, loss, accuracy, update_op], feed_dict={x: batch_data, y: batch_labels, step: i})
```

## Results
Hyper-parameter setting:
- batch size = 128
- dropout rate = 0.3 in the CNN layer and = 0.4 in the dense layer
- starting learning rate = 0.003
- decay step = 4000
- decay rate = 1.0/e

Iteration: 200, Loss: 15.1296, Accuracy: 0.9279

<img src="https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/iter200loss.png" width="550"/> <img src="https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/iter200acc.png" width="550"/>   

Iteration: 10000, Loss: 0.0163, Accuracy: 0.9917

<img src="https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/iter29550loss.png" width="550"/> <img src="https://github.com/lse-st449/lectures/raw/master/Week04/class/graphs/iter29550acc.png" width="550"/>   

## Homework
- Make the CNN work
- Try different hyper-parameter settings 
- Achieve 98% above accuracy
