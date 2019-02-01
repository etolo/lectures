"""Homework 2 Perceptron Learning Algorithm
   AI and Deep Learning
   LSE 2019
   Author: Tianlin Xu"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


def show_data(c1, c2, gamma):
    plt.scatter(c1[:, 0], c1[:, 1])
    plt.scatter(c2[:, 0], c2[:, 1])
    plt.axvline(x=gamma / 2.0, color='k', linestyle='--')
    plt.axvline(x=-gamma / 2.0, color='k', linestyle='--')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def plot_decision_boundary(data, labels, weights):
    xx, yy = np.mgrid[-1.1:1.1:0.1, -0.7:0.7:0.1]
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = predict(grid, weights)
    zz = np.array(preds).reshape(xx.shape)
    plt.contour(xx, yy, zz, levels=[0.01], cmap='winter')
    plt.scatter(data[:, 0], data[:, 1], s=20, cmap='cool', c=labels, vmin=0, vmax=1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def threshold_function(x):
    return tf.sign(tf.sign(x) + 0.00001)


def predict(data, weights):
    pred_1 = [np.inner(d, weights) for d in data]
    # y = threshold_function(pred_1)
    # return y.eval()
    return pred_1


def plot_loss(loss_list):
    plt.plot(loss_list)
    plt.title('Loss')
    plt.show()


def main():
    gamma = 0.1
    a1 = np.random.uniform(-1.0, -gamma / 2.0, size=200)
    a2 = np.random.uniform(-0.5, 0.5, size=200)

    x1 = np.random.uniform(gamma/2.0, 1.0, size=200)
    x2 = np.random.uniform(-0.5, 0.5, size=200)

    c1 = np.squeeze(np.dstack((a1, a2)))
    c2 = np.squeeze(np.dstack((x1, x2)))

    show_data(c1, c2, gamma)

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

    lr = 0.01

    # define the graph
    x = tf.placeholder(tf.float32, shape=[2, ])
    y = tf.placeholder(tf.float32, shape=())

    weights = tf.get_variable("weights", [2, ], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

    output = tf.reduce_sum(tf.multiply(weights, x))
    y_hat = threshold_function(output)

    y_times_y_hat = tf.multiply(y, y_hat)
    loss = tf.math.maximum(-y * output, 0)

    assign_op = None
    # weights += lr * y * x
    if y != y_hat:
        assign_op = tf.assign(weights, weights + lr * y * x)
    # optimiser = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    loss_list = []
    count = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        while True:
            print('Number of full pass:', count + 1)
            count += 1
            stopping = np.zeros(400)

            for i in range(len(data)):
                _, l, cond = sess.run([assign_op, loss, y_times_y_hat], feed_dict={x: data[i], y: labels[i]})
                loss_list.append(l)
                stopping[i] = cond

            # plot_loss(loss_list)
            all_greater = False if False in [item > 0.0 for item in stopping] else True
            if all_greater:
                print("Training completed")
                break

        w = weights.eval()

        plot_decision_boundary(data, labels, w)


if __name__ == '__main__':
    main()