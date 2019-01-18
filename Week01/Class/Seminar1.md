# Seminar 1: Getting started with TensorFlow

**Contents**:
* [Introduction to TensorFlow](#what-is-tensorflow) 
* [Introduction to computational graphs](#what-is-a-computational-graph)                 
* [Basic oprations in TensorFlow](#basic-operations)   
* [TensorFlow data type](#tensorflow-dtype-vs-numpy-dtype)       
* [Graph building and execution](#implementation-of-your-graph)                                       
* [Example: linear regression](#linear-regression) 
  
## What is TensorFlow?

TensorFlow is an open source Python library for machine learning research. Just like other libraries, such as Numpy and Scipy, TensorFlow is designed to make the implementation of a program easier and faster.

## TensorFlow architecture

![tf](https://github.com/lse-st449/lectures/raw/master/Week01/Class/graphs/tf_architecture.png)

All TensorFlow modules and classes can be found [here](https://www.tensorflow.org/api_docs/python/tf). 


#### Low level APIs and high level APIs

You can either code in the low-level TensorFlow API (**TensorFlow Core**), i.e., building computational graphs and execute them in a TensorFlow session, or in the high-level APIs using **Keras** and **Eager execution**. 

Although the high-level APIs provide a much simpler and consistent interface, it is still beneficial to learn how to use Tensorflow Core for the following reasons as mentioned in the official documents:

* Experimentation and debugging are both more straightforward when you can use low level TensorFlow operations directly.
* It gives you a mental model of how things work internally when using the higher level APIs.

#### Other options:

Theano, Pytorch

  
## What is a tensor?

A tensor is an n-dimensional array. A scalar is treated as a zero-dimensional tensor.

You can think it as a Python list or a Numpy array but with different data types. Let us demonstrate this by an example:


Firstly, import TensorFlow and Numpy:

```
import tensorflow as tf
import numpy as np
```

Define a 2 by 2 matrix using a Python list, a Numpy array and a tensor:

```
a1 = [[0.0, 0.5], [1.0, 0.0]]
a2 = np.array([[0.0, 0.5], [1.0, 0.0]], dtype=np.float32)
a3 = tf.constant([[0.0, 0.5], [1.0, 0.0]])

```

Print the data type of each object:

```
print(type(a1))
print(type(a2))
print(type(a3))
```

You will see the results as

```
<class 'list'>
<class 'numpy.ndarray'>
<class 'tensorflow.python.framework.ops.Tensor'>
```

## What is a computational graph?

For equation <img src="https://latex.codecogs.com/svg.latex?\Large&space;y=(a-b)+c*d" title="linear" />, the data flow shown below is called a computational graph:

<img  src="https://github.com/lse-st449/lectures/raw/master/Week01/Class/graphs/computational%20graphs.png"  width="650"  height="400">

Nodes are mathematical operations (e.g. `minus`, `mul`, and `add` in the graph), variables (e.g. `a`, `b`, `c` and `d`),  constants and tensors, i.e., your data.

  
## Implementation of your graph

#### Construction step

In our first seminar, the most important thing for you to learn about Tensorflow is that you start with **building the structure only**.


Let's now define:

```
a = tf.constant(3.0)
b = tf.constant(2.0)
c = tf.constant(1.0)
d = tf.constant(5.0)

a_minus_b = tf.subtract(a, b)
c_times_d = tf.multiply(c, d)
y = tf.add(a_minus_b, c_times_d)
```

What would happen if we simply print `y`?

```
print(y)
```

You will get:

```
Tensor("Add:0", shape=(), dtype=float32) #NOT a number 6!
```

This is because so far we have only built the structure of the algorithm.

#### Execution Step

To get the value of `y`, you will have to summon a **session** so you can launch the operations,

```
with tf.Session() as sess:
    print(sess.run(y))
```

`tf.Session()` encapsulates the environment where operations are executed and tensors are evaluated.

Session will also allocate memory to store the current values of variables.


If you see this warning,

![alt text](https://github.com/lse-st449/lectures/raw/master/Week01/Class/graphs/warning.PNG)

 
add the following at the beginning of your code

```
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
```

The warning states that your CPU doesn't support AVX.


## Basic operations

  

#### Constants

```
tf.constant(value, dtype=None, shape=None,name='Const',verify_shape=False)
```

Example:

```
tensor = tf.constant([1, 2, 3, 4])
```

#### Define a tensor with a specific value and shape

```
tf.zeros(shape, dtype=tf.float32, name=None)
```

Example:

```
x=tf.zeros([2, 3])

with tf.Session() as sess:
    print(sess.run(x))
```

Output tensor:

```
[[0. 0. 0.]
[0. 0. 0.]]
```

You can also create a tensor of the shape as another tensor but all elements in the new tensor are zeros.

```
tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)
```

Example:

```
input = [[0, 1],[2, 3]]
```

`tf.zeros_like(input)` will output

```
[[0, 0],
[0, 0]]
```

The following operations are also available

```
tf.ones(shape, dtype=tf.float32, name=None)
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)
```

or

```
tf.fill(dims, value, name=None)
```

Example:

`tf.fill([2, 3], 8)` will have an output tensor with shape `[2, 3]` and all elements of value `8`:

```
[[8, 8, 8],
[8, 8, 8]]
```

#### Generate a sequence of numbers

Generate a fixed number of values that are evenly-spaced from `start` to `stop`. The last number equals to `stop`.

```
tf.lin_space(start, stop, num, name=None)
```

Generate a sequence of numbers that begins at start and extends by increments of `delta` up to but not including `limit`.

```
tf.range(start, limit=None, delta=1, dtype=None, name='range')
```

#### Generate random constants

```
tf.random_normal
tf.truncated_normal
tf.random_uniform
tf.random_shuffle
tf.random_crop
tf.multinomial
tf.random_gamma
```

Fix a seed for replicability

```
tf.set_random_seed(seed)
```

#### Operations

##### Mathematical operations

```
tf.add
tf.subtract
tf.mul
tf.div
tf.exp
tf.log
...

```

  

##### Arithmetic operations

```
tf.abs
tf.negative
tf.sign
tf.square
tf.sqrt
...

```

  

##### Array operations

```
tf.concat
tf.slice
tf.split
tf.rank
tf.shape
...

```
##### Matrix operations

```
tf.matmul
tf.matrix_inverse
tf.matrix_determinant
...

```

#### Tensorflow dtype vs numpy dtype

TensorFlow takes Python natives types: boolean, numeric (int, float), and strings.

For the full list of TensorFlow data types, see [here](https://www.tensorflow.org/api_docs/python/tf/DType).

Let's compare TensorFlow with Numpy data type in Anaconda prompt under the environment where TensorFlow is installed,

![alt text](https://github.com/lse-st449/lectures/raw/master/Week01/Class/graphs/dtype.PNG)

As you can see, TensorFlow integrates seamlessly with Numpy types.

You can also pass Numpy types to TensorFlow Operations:

```
tf.zeros([1, 2], dtype=np.float32)
```

However, you should always use TensorFlow data type whenever possible. This is because

* TensorFlow has to calculate the type of the array and may even have to convert it to TensorFlow type, which could slow your code down

* Numpy arrays are not GPU compatible.

  

## tf.Variable

```
tf.Variable(<initial-value>, name=<optional-name>)
```

tf.Variable class holds several methods:

```
tf.Variable.initializer
tf.Variable.value
tf.Variable.assign
tf.Variable.assign_add
...

```

#### tf.constant vs tf.Variable

- `tf.constant` is an operation, whilst `tf.Variable` is a class with many operations.

- Constants are stored in the graph and every time when you load the graph the values created by `tf.constant` are replicated.

When the constant values are big, loading a graph becomes expensive. In comparison, variables are stored separately.

- You cannot change the value of a tensor if it's declared using `tf.constant`.

One can initialise a variable by

```
a = tf.Variable(2)
```

and change the original value with

```
a.assign(a + 1.0)
```

or

```
a.assign_add(1.0)
```

#### tf.Variable vs tf.get_variable

In practice, it's always recommended to use `tf.get_variable`:

```
tf.get_variable(name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=None, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, custom_getter=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, aggregation=tf.VariableAggregation.NONE)
```

The reasons why we prefer `tf.get_variable` are

* `tf.get_variable` also creates an instance of `tf.Variable` class

* It will make it easier to refactor your code if you need to share variables at any time; we will revisit this point after we explain the concept of a `scope` in TensorFlow.

Assuming that, in equation <img src="https://latex.codecogs.com/svg.latex?\Large&space;y=(a-b)+c*d" title="linear" />, `a`, `b`, `c` are constants and `d` is a parameter, let's rewrite our first implementation using `tf.get_variable`:

```
import tensorflow as tf

a = tf.constant(3.0)
b = tf.constant(2.0)
c = tf.constant(1.0)
d = tf.get_variable("d", initializer=tf.constant(5.0))

a_minus_b = tf.subtract(a, b)
c_times_d = tf.multiply(c, d)
y = tf.add(a_minus_b, c_times_d)
 
with tf.Session() as sess:
    print(sess.run(y))
```

If you run the code above, you will encounter an error:

```
FailedPreconditionError: Attempting to use uninitialized value a
[[{{node a/read}} = Identity[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"](a)]]
```

**You will have to initialise your variables,** i.e., the initializer in `tf.get_variable` needs to be executed within a session.

We update the code:

```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(y))
```

This is the most convenient way to initialise all variables at once. You can also initialise a single variable by

```
sess.run(a.initializer)
```

or a subset of variables by

```
sess.run(tf.variables_initializer([a, b]))
```

## tf.placeholder

`tf.placeholder` allocates storage for your data. When declaring a placeholder, you only need to specify the type and shape of the data you will feed to your model for execution.

```
tf.placeholder(dtype, shape=None, name=None)
```

`shape=()` will accept a scalar as value for placeholder.

`shape=None` indicates that a tensor of any shape will be accepted but it's problematic for debugging and many operations.

If you know the shape of your data, it's always a good practice to specify it in placeholder.

#### tf.Variable vs tf.placeholder

*  `tf.Variable` is for trainable variables such as weights, biases and other parameters for your model, whilst `tf.placeholder` is for training data.

*  `tf.Variable` needs initialisation while `tf.placeholder` doesn't.

* The values declared by `tf.Variable` will be modified during training, whilst data held by `tf.placeholder` remains unchanged.

#### Build the graph

If we declare one of the tensors in our code using `tf.placeholder`,

```
a = tf.placeholder(tf.float32, shape=())
b = tf.constant(2.0, dtype=tf.float32)
c = tf.constant(1.0, dtype=tf.float32)
d = tf.constant(5.0, dtype=tf.float32)

a_minus_b = tf.subtract(a, b)
c_times_d = tf.multiply(c, d)
y = tf.add(a_minus_b, c_times_d)

with tf.Session() as sess:
    print(sess.run(y))
```

You will see an error when running it:

```
InvalidArgumentError: You must feed a value for placeholder tensor 'Placeholder' with dtype float and shape [0]
[[{{node Placeholder}} = Placeholder[dtype=DT_FLOAT, shape=[0], _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]
```

Feed a value to placeholder using a dictionary with tensor `a` as the key:

```
with tf.Session() as sess:
    print(sess.run(y, feed_dict={a: 3.0}))
```

## Linear regression

We intend to model a linear relationship between a dependent variable `y` and an independent variable `x`.

Consider a simple linear regression model <img src="https://latex.codecogs.com/svg.latex?\Large&space;y=wx+b" title="linear" /> , where `w` is the weight parameter and `b` is the bias parameter.

Download `auto_insurance_in_sweden.csv` from the course repository.

### Phase 0: Data processing

#### Step 1: Load and separate the paired data

```
import tensorflow as tf
import numpy as np
import pandas as pd

# Load data to Google Colab from your computer
from google.colab import files
files.upload()
```

Choose a file from your computer and upload it. If successful, you will see

![alt text](https://github.com/lse-st449/lectures/raw/master/Week01/Class/graphs/upload_file.PNG)

Read data and get inputs `x` and `y` separately,

```
df = pd.read_csv('auto_insurance_in_sweden.csv', header=0)
x = df.iloc[:, 0].values # Independent variable
y = df.iloc[:, 1].values # Dependent variable
```

Visualise the raw data:

```
plt.scatter(x, y)
```

![alt text](https://github.com/lse-st449/lectures/raw/master/Week01/Class/graphs/raw_data.png)

### Phase 1: Building the graph

#### Step 2: Create placeholders for data

```
data_x = tf.placeholder(tf.float64)
data_y = tf.placeholder(tf.float64)
```

#### Step 3: Create variables for parameters

```
w = tf.get_variable(name="weights", shape=[], dtype=tf.float64, initializer=tf.random_normal_initializer())
b = tf.get_variable(name="bias", shape=[], dtype=tf.float64, initializer=tf.constant_initializer(0.0))
```

#### Step 4: Specify loss function
```
y_predicted = tf.add(tf.multiply(w, data_x), b) # tf.multiply is element-wise multiplication
# meam squared error
loss = tf.reduce_mean(tf.square(y - y_predicted))
```

#### Step 5: Select an optimiser
```
optimiser = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)
```
Sometimes you may need to modify your graph or learning rate for the optimiser after running the cell where you build it, you will run into this error:

![alt text](https://github.com/lse-st449/lectures/raw/master/Week01/Class/graphs/reuse_error.PNG)

To solve the problem, add

```
tf.reset_default_graph()
```
at the beginning of your code in this cell.

### Phase 2: Executing with a session

#### Step 6: Train the model

```
step = 500 # total number of iterations
with tf.Session() as sess:
sess.run(tf.global_variables_initializer())
loss_list = []
for i in range(step):
    _, l = sess.run([optimiser, loss], feed_dict={data_x: x, data_y: y})
    loss_list.append(l) # save the loss at each iteration for plot so you can visualise the training process
```

### Step 7: Plot the value of the loss function during training

```
p = np.arange(step)
plt.plot(p, loss_list)
```

![alt text](https://github.com/lse-st449/lectures/raw/master/Week01/Class/graphs/loss.png)

#### Step 8: Plot your results

By the end of training, you need to fetch the trained parameter values by adding

```
w_val, b_val = sess.run([w, b])
```
in your session.

Finally, plot the trained model,

```
plt.scatter(x, y)
y_trained = x * w_val + b_val
plt.plot(x, y_trained, 'r')
plt.show()
```

![alt text](https://github.com/lse-st449/lectures/raw/master/Week01/Class/graphs/trained.png)

## Homework: logistic regression
Use the following code to generate data:
```
x1 = np.random.normal(-4, 2, 1000)
x2 = np.random.normal(4, 2, 1000)
xs = np.append(x1, x2)
ys = np.asarray([0.] * len(x1) + [1.] * len(x2))
```
 
 Use the data `xs` and `ys` for training. The model is <img src="https://latex.codecogs.com/svg.latex?\Large&space;\hat{y}=\sigma(w*x+b)" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />, where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\sigma(\cdot)"/> is the sigmoid function and `w` and `b` are the parameters of the model. 
 
 **Task**: Find the optimal parameters and plot the fitted model to the data.
 
 **Hints**: 
 - You could use `tf.sigmoid()` function in TensorFlow
 - Use cross entropy loss: <img src="https://latex.codecogs.com/svg.latex?\Large&space;\mathcal{L}=-y\log\hat{y}-(1-y)\log(1-\hat{y})" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" /> 
 - Use **gradient decent optimiser** with learning rate parameter equal to 0.01 
 - Use the number of iterations equal to 1000 


## References

* [Official TensorFlow tutorial](https://www.tensorflow.org/tutorials/)
* [Offical TensorFlow guide for low level APIs](https://www.tensorflow.org/guide/low_level_intro)
* [Offical TensorFlow guide for high level APIs](https://www.tensorflow.org/guide/keras)
