# Seminar 3: Optimization for training neural networks

Our goals in this seminar session are:
* Explain gradient computation in TensorFlow
* Learn about how to use different optimizers in TensorFlow
* Homework: evaluate gradient descent algorithm with random initalization for a non-convex optimisation problem


### Get Gradients in TensorFlow
```
tf.gradients(ys, xs, grad_ys=None, name='gradients')
```
This method constructs symbolic derivatives of sum of `ys` w.r.t. `x` in `xs`.
For example:
```
dw, db = tf.gradients(loss, [w, b])
```
This returns the gradients of the loss function w.r.t parameter `w` and `b`. 

`grad_ys` is a list of tensors of the same length as `ys` that holds the initial gradients for each `y` in `ys`. When `grad_ys` is None, we fill in a tensor of '1's of the shape of `y` for each `y` in `ys`. A user can provide their own initial `grad_ys` to compute the derivatives using a different initial gradient for each `y` (e.g., if one wanted to weight the gradient differently for each value in each `y`).

## Optimizers in TensorFlow

### Gradient Descent (GD)

GD update: 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;w^{(t+1)}=w^{(t)}-\eta\bigtriangledownf(w^{(t)})" title="\Large GD" />, 
where <img src="https://latex.codecogs.com/svg.latex?\Large&space;\eta" title="\Large GD" /> is the step size (learning rate) and <img src="https://latex.codecogs.com/svg.latex?\Large&space;\bigtriangledownf(w^{(t)})" title="\Large GD" /> is the gradient of function <img src="https://latex.codecogs.com/svg.latex?\Large&space;f" title="\Large GD" /> at point <img src="https://latex.codecogs.com/svg.latex?\Large&space;w^{(t)}" title="\Large GD" />. In TensorFlow, this algorithm is implemented in the `tf.train.GradientDescentOptimizer` class. The constructor of the class is
```
tf.train.GradientDescentOptimizer(learning_rate, use_locking=False, Name='GradientDescent')
```

The commonly used methods of this class are: `.compute_gradients()`, `.apply_gradient()` and `.minimize()`. 
```
compute_gradients(loss, var_list=None, gate_gradients=GATE_OP, aggregation_method=None, colocate_gradients_with_ops=False, grad_loss=None)
```

This method wraps the `tf.get_gradient()` function with additional settings. The `gate_gradients` argument is explained in [TensorFlow-optimizer.py implementation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/optimizer.py#L281):
>`GATE_NONE`: Compute and apply gradients in parallel.  This provides the maximum parallelism in execution, at the cost of some non-reproducibility  in the results.  For example the two gradients of `matmul` depend on the input  values: With `GATE_NONE` one of the gradients could be applied to one of the  inputs _before_ the other gradient is computed resulting in non-reproducible  results.  

>`GATE_OP`: For each Op, make sure all gradients are computed before they are used. This prevents race conditions for Ops that generate gradients for multiple inputs where the gradients depend on the inputs. 

>`GATE_GRAPH`: Make sure all gradients for all variables are computed before any one of them is used. This provides the least parallelism but can be useful if you want to process all gradients before applying any of them. 


`.apply_gradient(grads_and_vars, global_step=None, name=None)` applies gradients to variables. 

```
minimize(loss, global_step=None, var_list=None, gate_gradients=GATE_OP, aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None)
```

This method combines `compute_gradients()` and `apply_gradients()`. As stated in the [official document](https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer#minimize)
>If you want to process the gradient before applying them call `compute_gradients()` and `apply_gradients()` explicitly instead of using this function.

Typically, you will construct a new optimizer and choose to minimize a loss function in the following way 

```
optimiser = tf.train.GradientDescentOptimiser(learning_rate=1e-3)
optimiser.minimize(some_loss_function)
```

### Momentum
```
tf.train.MomentumOptimizer(learning_rate, momentum, use_locking=False, name='Momentum', use_nesterov=False)
```
This function is an implementation of the momentum algorithm:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;v^{(t+1)}=mv^{(t)}+\eta\bigtriangledownf(w^{(t)})" title="\Large GD"/>,  
<img src="https://latex.codecogs.com/svg.latex?\Large&space;w^{(t+1)}=w^{(t)}+v^{(t+1)}" title="\Large GD" /> , 

where `m` is momentum and `eta` is the learning rate parameter.  

When the argument `use_nesterov` is set `True`,  the optimizer uses Sutskever's Nesterov momentum (2013), i.e., 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;v^{(t+1)}=w^{(t)}-\eta\bigtriangledownf(w^{(t)})" title="\Large GD" />,  
<img src="https://latex.codecogs.com/svg.latex?\Large&space;w^{(t+1)}=(1-m^{(t)})v^{(t+1)}+m^{(t)}v^{(t)}" title="\Large GD" /> 

### RMSProp 
```
tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, centered=False, name='RMSProp')
```

### Adagrad
```
tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.1, use_locking=False, name='Adagrad')
```
### Adaptive Moment Estimator (Adam)
```
tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
```

## Stochastic gradient descent algorithms

Stochastic gradient descent algorithms use a stochastic gradient vector computed by evaluating the gradient of the loss function for a random sample of training examples. This sample is referred to as a **mini-batch**. Each time the algorithm makes a pass through the whole training dataset, an **epoch** is completed. 

Continuing with the linear regression exercise example, 
```
df = pd.read_csv('auto_insurance_in_sweden.csv', header=0)  
zipped_xy = df.values
```
We obtain the paired `x` and `y` values that we can shuffle `x` with corresponding `y` at the same time. 
The next block of code should be written in the loop within `tf.Session()` to generate a random sample at each iteration step,
```
n_batches = np.floor(len(x)/batch_size).astype(int)
for i in range(epoch):  
    shuffled_xy = np.random.permutation(zipped_xy)  
    shuffled_x = shuffled_xy[:, 0]  
    shuffled_y = shuffled_xy[:, 1]  
  
    for n in range(n_batches):  
        batch_x = shuffled_x[n * batch_size:(n + 1) * batch_size]  
        batch_y = shuffled_y[n * batch_size:(n + 1) * batch_size]  
        _, l = sess.run([optimiser1, loss], feed_dict={data_x: batch_x, data_y: batch_y})
```
**Note**: Instead of feeding the entire dataset, we feed the mini-batches to placeholders.

#### Comparison of Adam algorithm with mini-batch Adam algorithm

Loss function using Adam algorithm compared to the loss function using mini-batch Adam algorithm, 

<img src="https://github.com/lse-st449/lectures/raw/master/Week03/class/graphs/loss_adam_wihtout_minibatch.png" width="550"/> <img src="https://github.com/lse-st449/lectures/raw/master/Week03/class/graphs/loss_stochastic_adam.png" width="550"/> 

Gradient trajectory using Adam algorithm compared with the gradient trajectory using mini-batch Adam algorithm with mini-batch size `15`, 

<img src="https://github.com/lse-st449/lectures/raw/master/Week03/class/graphs/grad_path_adam_wihtout_minibatch.png" width="550"/> <img src="https://github.com/lse-st449/lectures/raw/master/Week03/class/graphs/grad_path_stochastic_adam.png" width="550"/> 

## Monitor the optimization process

When the number and dimensionality of the parameters are both small, one can visualize the gradient trajectories as we have shown. However, this becomes impossible when either the number or dimensionality of parameters becomes large. Another approach is to visualize the loss curves.  

The following plot compares the performances of different optimizers in solving the logistic regression problem in homework 1:

![alt text](https://github.com/lse-st449/lectures/raw/master/Week03/class/graphs/loss_curve_comparison.png)

### How learning may go wrong

The plots in this section show how learning can go wrong due to either the gradient step being too small or too large or your optimizer being stuck at a local minimum. You may want to adjust the learning rate, change the initialization or try another optimizer.  

The examples are generated using the linear regression exercise in Seminar 1. 

### step size too small
When the learning rate is set to be `1e-8`, we do not observe convergence in the loss or gradient plot even after 50,000 iterations.   

<img src="https://github.com/lse-st449/lectures/raw/master/Week03/class/graphs/GD_lr-8.png" width="550"/> <img src="https://github.com/lse-st449/lectures/raw/master/Week03/class/graphs/GD_lr-8_grad.png" width="550"/> 

The gradient trajectory also shows slow movements towards the optimal point.  

<img src="https://github.com/lse-st449/lectures/raw/master/Week03/class/graphs/GD_lr-8_path.png" width="550"/> 

### Step size too large

When we increase the learning rate to `1e-2`, the gradients exploded and loss went to infinity. 

<img src="https://github.com/lse-st449/lectures/raw/master/Week03/class/graphs/GD_lr-2.png" width="550"/> <img src="https://github.com/lse-st449/lectures/raw/master/Week03/class/graphs/GD_lr-2_grad.png" width="550"/> 

<img src="https://github.com/lse-st449/lectures/raw/master/Week03/class/graphs/GD_lr-2_path.png" width="550"/> 

You may want to lower the learning rate or try a more robust optimizer.  

### Appropriate step size

After adjusting the learning rate to `1e-5`, we observe that both the loss and gradients converged after less than 200 iterations.

<img src="https://github.com/lse-st449/lectures/raw/master/Week03/class/graphs/GD_lr-5.png" width="550"/> <img src="https://github.com/lse-st449/lectures/raw/master/Week03/class/graphs/GD_lr-5_grad.png" width="550"/> 


<img src="https://github.com/lse-st449/lectures/raw/master/Week03/class/graphs/GD_lr-5_path.png" width="550"/> 


## Homework: Non-convex optimization

Consider loss function <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(x_1,x_2)=\frac{1}{2}x_1^2-\frac{1}{2}x^2_2(1-\frac{1}{2}x^2_2)" title="\Large GD"/>, 
- Compute critical points
- Qualify critical points as either local minima or saddle points, if there are any saddle points are they strict saddle points?
- Implement GD algorithm to minimize the loss function in TensorFlow
- Run your GD algorithm with learning_rate =  0.01 for different initial values given as follows:
  - (a, 0), where a is an arbitrary real value
  - (0, 0.5)
  - (0, -0.5)
  - (5, 0)
  - (5, 0.1)
  - (0, 10)
- Plot the parameter values versus the number of iterations. What do you observe? Comment.
- Run the algorithm for 10 different initial values taken uniformly at random from [-10, 10] x [-10, 10]
  -  Check the trajectories of parameter vector <img src="https://latex.codecogs.com/svg.latex?\Large&space;(x_1,x_2)" title="\Large GD" /> versus the number of iterations
  -   Comment what you observe

**Hint**:

The gradient trajectory plots can be created by 
```
def loss_for_plot(x1, x2):  
    x1 = np.atleast_3d(np.asarray(x1))  
    x2 = np.atleast_3d(np.asarray(x2))  
    f = function(x1, x2)
    return f
    
def plot_cont_grad_path(true_x1, true_x2, grad_x1, grad_x2):
    n = 50

    x1_ = np.linspace(-3.0, 5.5, n)
    x2_ = np.linspace(-3.0, 4.0, n)

    z = loss_for_plot(x1_[np.newaxis, :, np.newaxis], x2_[:, np.newaxis, np.newaxis])
    z = np.squeeze(z)

    x_grid, y_grid = np.meshgrid(x1_, x2_)

    # A labeled contour plot for the cost function - marginal_llh
    plt.figure()
    level = np.arange(0.1, 100.0, 5.0)
    con = plt.contour(x_grid, y_grid, z, level)
    plt.clabel(con)
    # the target parameter values indicated on the contour plot using the true parameter values
    plt.scatter(true_x1, true_x2, c='r')

    # Plot the path of parameter chosen and arrows indicating the gradient direction
    plt.scatter(grad_x1[0], grad_x2[0], c='g')
    for j in range(1, 500):
        if j != 0 and j % 100 == 0:
            plt.scatter(grad_x1[j], grad_x2[j], c='g')
            plt.annotate('', xy=(grad_x1[j], grad_x2[j]), xytext=(grad_x1[j - 100], grad_x2[j - 100]),
                         arrowprops=dict(arrowstyle="->", facecolor='green'))
    plt.title('Gradients Path with learning rate=1e-2')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
```
