#!/usr/bin/env python

# originally from:
# https://github.com/sebastianheinz/stockprediction/blob/master/02_code/stockprediction.py
# https://medium.com/mlreview/a-simple-deep-learning-model-for-stock-price-prediction-using-tensorflow-30505541d877

import matplotlib

# workaround for Docker containers
matplotlib.use('Agg')

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from cassandra.io.libevreactor import LibevConnection
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy


### BULK OF MY EDITS START HERE

def get_session():
    '''
    Connect onto a Cassandra cluster with the driver.
    :return: A Cassandra session object for cluster interactions.
    '''

    # grab the cluster information using Docker-provided enviornmental variables
    CASSANDRA_HOST = os.environ['CASSANDRA_HOST']
    CASSANDRA_DC = os.environ['CASSANDRA_DC']

    # create a cluster object that will only connect to a single data center
    cluster = Cluster([CASSANDRA_HOST],
                      load_balancing_policy=DCAwareRoundRobinPolicy(
                          local_dc=CASSANDRA_DC), )

    # use the faster event loop provider
    cluster.connection_class = LibevConnection

    # create the Cassandra session for cluster interaction
    session = cluster.connect()

    # Panda-centric row factory
    def pandas_factory(colnames, rows):
        return pd.DataFrame(rows, columns=colnames)

    # use Panda-centric settings
    session.row_factory = pandas_factory
    session.default_fetch_size = None

    return session


# create a new Cassandra connection
session = get_session()

# grab all coin names with historical data
query = "SELECT DISTINCT coin FROM crypto.historical"
results = session.execute(query, timeout=None)
coin_results = results._current_rows

# populate the first DataFrame with Bitcoin price data
query = "SELECT timestamp, value FROM crypto.historical WHERE coin = '%s'"
new_query = query % 'bitcoin'
results = session.execute(new_query, timeout=None)
data = results._current_rows

# rename the column names
data.columns = ['timestamp', 'bitcoin']

# find the number of data points we will be looking for
valid_rows = len(data.index)

# remove Bitcoin from the coins list to avoid a duplicate DataFrame column
coin_results = coin_results[coin_results.coin != 'bitcoin']

# future queue for pending reads
futures = []

# loop through all known coins and asynchronously request data
for coin in coin_results['coin']:
    # grab historical data from Cassandra
    future = session.execute_async(query % coin, timeout=None)
    futures.append((coin, future))

# process the pending future
for coin, future in futures:
    results = future.result()

    # format Cassandra's return request
    c1 = results._current_rows
    c1.columns = ['timestamp', coin]

    # ensure that when merged, data is not lost
    tmp_data = pd.merge(data, c1, how='inner', on=['timestamp'])
    if len(tmp_data.index) < valid_rows:
        continue

    # if all rows are kept, proceed with the merge
    data = pd.merge(data, c1, how='inner', on=['timestamp'])

# show a sample of the data retrieved
print data

# remove the timestamp column which was used during our join phases
data = data.drop(['timestamp'], 1)

### BULK OF MY EDITS STOP HERE

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]

# Make data a np.array
data = data.values

# Training and test data
train_start = 0
train_end = int(np.floor(0.8 * n))
test_start = train_end + 1
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# Scale data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# Build X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# Number of stocks in training data
n_stocks = X_train.shape[1]

# Neurons
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128

# Session
net = tf.InteractiveSession()

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg",
                                                     distribution="uniform",
                                                     scale=sigma)
bias_initializer = tf.zeros_initializer()

# Hidden weights
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output weights
W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

# Output layer (transpose!)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Init
net.run(tf.global_variables_initializer())

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)
plt.show()
fig.savefig('result.png')

# Fit neural net
batch_size = 256
mse_train = []
mse_test = []

# Run
epochs = 100
for e in xrange(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 50) == 0:
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            plt.pause(0.01)
