import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.INFO)
np.random.seed(1)


def plot_multiple_accuracy(acc1, acc2, acc3, acc4):
    plt.plot(acc1, 'r-', label='GD')
    plt.plot(acc2, 'y-', label='AdaGrad')
    plt.plot(acc3, 'c-', label='Momentum=0.7')
    plt.plot(acc4, 'm-', label='Adam')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.title('Accuracy vs Epoches')
    plt.legend()
    plt.show()


def plot_multiple_loss(loss1, loss2, loss3, loss4):
    plt.plot(loss1, 'r-', label='GD')
    plt.plot(loss2, 'y-', label='AdaGrad')
    plt.plot(loss3, 'c-', label='Momentum=0.7')
    plt.plot(loss4, 'm-', label='Adam')
    plt.title('Loss vs Epoches')
    plt.ylabel('Loss')
    plt.xlabel('Training Epoches')
    plt.legend()
    plt.show()


def main(arg):
    mnist = input_data.read_data_sets("mnist", one_hot=False)
    test_set = mnist.test.images
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # define hyper-parameters
    dims = 784
    num_labels = 10
    learning_rate = 0.001
    batch_size = 128
    epochs = 45
    n_batches = np.floor(55000. / batch_size).astype(int)
    lam = 0.01
    init_w = tf.constant(np.random.randn(dims, num_labels) * 0.1, dtype=tf.float32)
    init_b = tf.constant(np.random.randn(num_labels) * 0.1, dtype=tf.float32)

    # Define the computational graph
    with tf.variable_scope('main', reuse=tf.AUTO_REUSE):
        train_dataset = tf.placeholder(tf.float32, shape=(batch_size, dims))
        train_labels = tf.placeholder(tf.int32, shape=(batch_size,))
        steps = tf.placeholder(tf.float32)

        weights = tf.get_variable(name='w', initializer=init_w, dtype=tf.float32)
        biases = tf.get_variable(name='b', initializer=init_b, dtype=tf.float32)

        logits = tf.add(tf.matmul(train_dataset, weights), biases)
        unregularized_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels,
                                                                                           logits=logits))
        regularizer = tf.nn.l2_loss(weights)
        loss = unregularized_loss + lam * regularizer

        new_lr = learning_rate / tf.sqrt(steps)

        # Optimizers
        optimizer1 = tf.train.GradientDescentOptimizer(new_lr).minimize(loss)
        optimizer2 = tf.train.AdagradOptimizer(new_lr).minimize(loss)
        optimizer3 = tf.train.MomentumOptimizer(new_lr, momentum=0.7, use_nesterov=True).minimize(loss)
        optimizer4 = tf.train.AdamOptimizer(new_lr).minimize(loss)

        test_prediction = tf.nn.softmax(tf.add(tf.matmul(test_set, weights), biases))

        predictions_idx = tf.argmax(test_prediction, axis=1)
        accuracy_test, update_op = tf.metrics.accuracy(labels=test_labels, predictions=predictions_idx)

    loss_list = []
    accuracy = []

    for op in [optimizer1, optimizer2, optimizer3, optimizer4]:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            print("Optimiser Initialized")

            for epoch in range(epochs):
                epoch_loss = []
                epoch_acc = []
                for batch in range(n_batches):
                    batch_data, batch_labels = mnist.train.next_batch(batch_size)

                    _, l, acc, _ = sess.run([op, loss, accuracy_test, update_op], feed_dict={train_dataset: batch_data,
                                                                                             train_labels: batch_labels,
                                                                                             steps: epoch + 1.0})
                    epoch_loss.append(l)
                    epoch_acc.append(acc)

                loss_list.append(np.mean(epoch_loss))
                accuracy.append(np.mean(epoch_acc))

    loss_list = np.reshape(loss_list, (-1, epochs))
    accuracy = np.reshape(accuracy, (-1, epochs))
    plot_multiple_loss(loss_list[0, :], loss_list[1, :], loss_list[2, :], loss_list[3, :])
    plot_multiple_accuracy(accuracy[0, :], accuracy[1, :], accuracy[2, :], accuracy[3, :])


if __name__ == '__main__':
    tf.app.run()