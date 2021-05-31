import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# import urllib


from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


'''

DESCRIPTION

    This file applies MLP model to SWAN inputs and outputs.

AUTHOR

    Fearghal O'Donncha <feardonn@ie.ibm.com>, IBM Research, Dublin, Ireland

COPYRIGHT

    International Business Machines Corporation (IBM).  All rights reserved.


Run this code as:
    python mlp_wave_model_tf.py

Function:
    This code is used to train and evaluate ability of MLP model
    to replicate SWAN model on high resolution dataset

Things to note:

Files Required
    1) xdt1.txt & ydt1.txt; if not present then downloaded according
       to the instructions in README file
/
'''

tf.logging.set_verbosity(tf.logging.INFO)

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x,weights['h1']), biases['b1']))
    # Hidden layer with RELU activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Hidden layer with RELU activation
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    # Output layer with linear activation
    out_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
    return out_layer


def run():
    #read from X, Y txt
    datadirectory = 'Data/'

    # num_epochs = 1500  # not sure on this
    num_epochs = 1500  # not sure on this
    batch_size = 128
    DISPLAY_STEP = 50

    # Network Parameters
    nlayers = 3
    beta = 0.0001
    # nneurons = 20
    nneurons = 20
    print("number of neurons:", nneurons)
    n_hidden_1 = nneurons  # 1st layer number of neurons
    n_hidden_2 = nneurons  # 2nd layer number of neurons
    n_hidden_3 = nneurons  # 2nd layer number of neurons
    # learning rate
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               10000, 0.96, staircase=True)


    #############################
    # Load training & eval data #
    #############################
    ## check if file exists and if not download from box.com folder
    x_fname = datadirectory + 'xdt1.txt'
    y_fname = datadirectory + 'ydt1.txt'
    X = np.genfromtxt(x_fname, delimiter="\t")
    Y1 = np.loadtxt(y_fname, delimiter="\t")[:, 0:-1]
    N_INSTANCES = X.shape[0]  # Number of instances
    N_INPUT = X.shape[1]   # Input size
    N_CLASSES = Y1.shape[1]  # Number of classes (output size)
    TEST_SIZE = 0.1  # Test set size (% of train set)
    TRAIN_SIZE = int(N_INSTANCES * (1 - TEST_SIZE))  # Train size
    RANDOM_STATE = 100

    print('Total Data Sets available: \t' + str(len(X)))
    # Train-test split
    scaler = preprocessing.StandardScaler()
    X1 = scaler.fit_transform(X)  # normalize the data
    data_train, data_test, labels_train, labels_test = train_test_split(X1,
                                                                        Y1,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)

    print("Variables loaded successfully...\n")
    print("Number of predictors \t%s" % (N_INPUT))
    print("Number of classes \t%s" % (N_CLASSES))
    print("Number of instances \t%s" % (N_INSTANCES))
    print("\n")
    ###############
    # Graph input #
    ###############
    with tf.name_scope('io'):
        # tf Graph input
        X = tf.placeholder("float64", [None, N_INPUT])
        Y = tf.placeholder("float64", [None, N_CLASSES])

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([N_INPUT, n_hidden_1], dtype=tf.float64)),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], dtype=tf.float64)),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], dtype=tf.float64)),
            'out': tf.Variable(tf.random_normal([n_hidden_3, N_CLASSES], dtype=tf.float64))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1], dtype=tf.float64)),
            'b2': tf.Variable(tf.random_normal([n_hidden_2], dtype=tf.float64)),
            'b3': tf.Variable(tf.random_normal([n_hidden_3], dtype=tf.float64)),
            'out': tf.Variable(tf.random_normal([N_CLASSES], dtype=tf.float64))
        }

    ###################
    # Construct model #
    ###################



    pred = multilayer_perceptron(X, weights, biases)

    # Define loss and optimizer
    regularizer = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2'])  + tf.nn.l2_loss(weights['h3'])
    cost = tf.reduce_mean(tf.square(pred - Y))  + (beta*regularizer)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)


    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


    with tf.Session() as sess:
        # Initialize the variables (like the epoch counter).
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #    sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Training loop
        for epoch in range(num_epochs):
            avg_cost = 0.
            total_batch = int(data_train.shape[0] / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                randidx = np.random.randint(int(TRAIN_SIZE), size=batch_size)
                batch_xs = data_train[randidx, :]
                batch_ys = labels_train[randidx, :]
                sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys}) / total_batch

                # Display logs per epoch step
            if epoch % DISPLAY_STEP == 0:
                print("Epoch: %03d/%03d cost: %.9f" % (epoch, num_epochs, avg_cost))

        print("End of training.\n")
        print("Testing...\n")
        # ------------------------------------------------------------------------------
        # Testing

        test_acc = sess.run(cost, feed_dict={X: data_test, Y: labels_test})
        print("Test accuracy: %.3f" % (test_acc))

        coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

    print("Session closed!")

if __name__ == '__main__':
    run()
