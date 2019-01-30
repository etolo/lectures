"""Homework 1
   AI and Deep Learning
   LSE 2019
   Author: Tianlin Xu"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
np.random.seed(1)


def plot_loss(loss):
    p = np.arange(len(loss))
    plt.plot(p, loss)
    plt.title('Loss')
    plt.show()


def plot_prediction(xs, ys, w_val, b_val):
    all_xs = np.linspace(-10., 10., 100)
    with tf.Session() as sess:
        predicted_vals = tf.sigmoid(all_xs * w_val + b_val).eval()
    plt.plot(all_xs, predicted_vals, 'r')
    plt.scatter(xs, ys)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def main():
    x1 = np.random.normal(-4, 2, 1000)
    x2 = np.random.normal(4, 2, 1000)
    xs = np.append(x1, x2)
    ys = np.asarray([0.] * len(x1) + [1.] * len(x2))

    learning_rate = 0.01
    n_iterations = 1000

    plt.scatter(xs, ys)
    plt.show()

    x = tf.placeholder(tf.float32, shape=len(xs))
    y = tf.placeholder(tf.float32, shape=len(ys))
    w = tf.get_variable(name='w', shape=[], initializer=tf.constant_initializer(0.5))
    b = tf.get_variable(name='b', shape=[], initializer=tf.constant_initializer(-1.0))
    y_hat = tf.sigmoid(tf.add(tf.multiply(x, w), b))
    cost = tf.reduce_mean(-y * tf.log(y_hat) - (1 - y) * tf.log(1 - y_hat))

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    loss_list = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_iterations):
            _, loss = sess.run([train_op, cost], {x: xs, y: ys})
            loss_list.append(loss)
        w_val, b_val = sess.run([w, b], {x: xs, y: ys})

    plot_loss(loss_list)
    plot_prediction(xs, ys, w_val, b_val)


if __name__ == '__main__':
    main()

