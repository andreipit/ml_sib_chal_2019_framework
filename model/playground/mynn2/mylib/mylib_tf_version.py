print('x_train',x_train.shape,'; y_train =',y_train.shape,'; x_test',x_test.shape,'; y_test',y_test.shape,'; x_val',x_val.shape,'; y_val',y_val.shape)

# SIGMOID TF FOR 2 CLASSES 2or3 FEATURES, WITHOUT ONE HOT Y, WORKS WITH DICT AND WITH FILE

# SOFTMAX NOT WORKING


import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
#from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
%matplotlib inline
np.random.seed(1)

# ass data shapes: # X_train shape: (12288, 1080) # Y_train shape: (6, 1080) # X_test shape: (12288, 120) # Y_test shape: (6, 120)
# x_train (2, 12) ; y_train = (1, 12) ; x_test (2, 12) ; y_test (1, 12) ; x_val (2, 6) ; y_val (1, 6)
# model = my_l_layers_ffn(optimizer='adam', DO_keep_prob=1, L2_lambd=0, layers_dims=[x_train.shape[0],x_train.shape[0]*10,1], num_epochs = 2000, learning_rate = 0.0003, X=x_train, Y=y_train, print_cost = True,  activation = 'relu', activation_l = 'sigmoid', loss='log', mini_batches=mini_batches)
#def model(X_train, Y_train, X_test=None, Y_test=None, learning_rate = 0.0001, num_epochs = 200, minibatch_size = 32, print_cost = True):
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0003, num_epochs = 2000, minibatch_size = 4, print_cost = True, softmax=True):
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # def create_placeholders(n_x, n_y): 
    X = tf.placeholder(dtype=tf.float32, shape=[n_x, None], name="X") 
    Y = tf.placeholder(dtype=tf.float32, shape=[n_y, None], name="Y")
    
    # def initialize_parameters():
    # for images xavier: W1,b1,W2,b2,W3,b3 =  tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1)), tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer()), tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1)), tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer()), tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1)), tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())
    # for fruits xavier: W1,b1,W2,b2,W3,b3 =  tf.get_variable("W1", [25, 2], initializer=tf.contrib.layers.xavier_initializer(seed=1)), tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer()), tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1)), tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer()), tf.get_variable("W3", [1, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1)), tf.get_variable("b3", [1, 1], initializer=tf.zeros_initializer())
    l_dims=[25,12,1]
    W1,b1 = tf.get_variable("W1", [l_dims[0], n_x],       initializer=tf.contrib.layers.variance_scaling_initializer(seed=1)),  tf.get_variable("b1", [l_dims[0], 1], initializer=tf.zeros_initializer())
    W2,b2 = tf.get_variable("W2", [l_dims[1], l_dims[0]], initializer=tf.contrib.layers.variance_scaling_initializer(seed=1)),  tf.get_variable("b2", [l_dims[1], 1], initializer=tf.zeros_initializer())
    W3,b3 = tf.get_variable("W3", [l_dims[2], l_dims[1]], initializer=tf.contrib.layers.variance_scaling_initializer(seed=1)),  tf.get_variable("b3", [l_dims[2], 1], initializer=tf.zeros_initializer())
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    
    # def forward_propagation(X, parameters):
    Z1 = tf.add(tf.matmul(W1, X), b1); A1 = tf.nn.relu(Z1); Z2 = tf.add(tf.matmul(W2, A1), b2); A2 = tf.nn.relu(Z2); Z3 = tf.add(tf.matmul(W3, A2), b3) # Z3 = np.dot(W3,Z2) + b3

    # def compute_cost(Z3, Y)
    if softmax: cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(Z3), labels= tf.transpose(Y))) # Add cost function to tensorflow graph
    else:       cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(Z3), labels= tf.transpose(Y))) # Add cost function to tensorflow graph

    # Backpropagation
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    
    init = tf.global_variables_initializer() # Initialize all the variables

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init) # Run the initialization
        
        for epoch in range(num_epochs): # Do the training loop
            
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch # Select a minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            if print_cost == True and epoch % 100 == 0: print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0: costs.append(epoch_cost)
        # plot the cost
        plt.plot(np.squeeze(costs)); plt.ylabel('cost'); plt.xlabel('iterations (per tens)'); plt.title("Learning rate =" + str(learning_rate)); plt.show()
        parameters = sess.run(parameters); print ("Parameters have been trained!") # lets save the parameters in a variable
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y)) # Calculate the correct predictions
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # Calculate accuracy on the test set
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train})); print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        return parameters
    
#parameters = model(softmax=SOFTMAX, X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test) # Expected Output:  **Train Accuracy** 0.999074  **Test Accuracy** 0.716667
