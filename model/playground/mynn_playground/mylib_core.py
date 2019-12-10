import numpy as np
def speed(tic): 
    import time
    return str(1*(time.time()-tic)) + "s" # tic = time.time() ... print(speed(tic))

def speed_sec(tic): 
    import time
    #return str(1*(time.time()-tic)) + "s" # tic = time.time() ... print(speed(tic))
    return 1*(time.time()-tic)

def gradient_checking(X, Y, wb_params, gradients, num_i, L, m, L2_lambd, DO_keep_prob, forward_fun, compute_cost_fun): #  doesnt work with dropout
    def dictionary_to_vector(parameters):
        keys = []; count = 0
        #for key in ["W1", "b1", "W2", "b2", "W3", "b3"]: # L = 3  layers_dims=[3,5,3,1]
        keys = []
        for i in range(1, L + 1): keys.append('W'+str(i)); keys.append('b'+str(i)) # #for i in range(1,4) => i: 1 2 3
        for key in keys: #for key in list(parameters.keys()): # badway, cause depends from dictionary order
            new_vector = np.reshape(parameters[key], (-1,1)) # flatten parameter
            keys = keys + [key]*new_vector.shape[0]
            if count == 0: theta = new_vector
            else: theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1
        return theta, keys
    def vector_to_dictionary(theta, p):
        parameters = {}
        head = 0;  # Example for L=3 layers_dims=[3,5,3,1]:   parameters["W1"] = theta[:15].reshape(p['W1'].shape);parameters["b1"] = theta[15:20].reshape(p['b1'].shape);parameters["W2"] = theta[20:35].reshape(p['W2'].shape);parameters["b2"] = theta[35:38].reshape(p['b2'].shape);parameters["W3"] = theta[38:41].reshape(p['W3'].shape);parameters["b3"] = theta[41:42].reshape(p['b3'].shape)
        for i in range(1, L + 1): #for i in range(1,4) => i: 1 2 3
            matrix_len = p["W"+str(i)].shape[0] * p["W"+str(i)].shape[1]
            parameters["W"+str(i)] = theta[head:head+matrix_len].reshape(p['W'+str(i)].shape) #;print('Ws',head,head+matrix_len) # Ws 0 15   Ws 20 35   Ws 38 41
            head += matrix_len
            matrix_len = p["b"+str(i)].shape[0] * p["b"+str(i)].shape[1]
            parameters["b"+str(i)] = theta[head:head+matrix_len].reshape(p['b'+str(i)].shape)# ;print('bs',head,head+matrix_len) # bs 15 20     bs 35 38     bs 41 42
            head += matrix_len 
        return parameters
    def gradients_to_vector(gradients):
        count = 0; keys = [] #for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]: # for L=3 layers_dims=[3,5,3,1]
        for i in range(1, L + 1): keys.append('dW'+str(i)); keys.append('db'+str(i)) # #for i in range(1,4) => i: 1 2 3
        for key in keys: #for key in list(reversed(list(gradients.keys()))): # badway: depends on order and wrong return # ['db1', 'dW1', 'db2', 'dW2', 'db3', 'dW3'] 
            new_vector = np.reshape(gradients[key], (-1,1)) # flatten parameter
            if count == 0: theta = new_vector
            else: theta = np.concatenate((theta, new_vector), axis=0)
            count = count + 1
        return theta
    def flatten_dict_of_matrices(dict_of_matrices):
        list_of_lists = [list(item) for sublist in dict_of_matrices.values() for item in sublist]; 
        return [item for sublist in list_of_lists for item in sublist]
    wb_p_values, p_keys_useless = dictionary_to_vector(wb_params) # print(p_values.shape) # (47, 1) #grad = gradients_to_vector(gradients) # print(p_keys_useless) # ['W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'W1', 'b1', 'b1', 'b1', 'b1', 'b1', 'W2', 'W2', 'W2', 'W2', 'W2', 'W2', 'W2', 'W2', 'W2', 'W2', 'W2', 'W2', 'W2', 'W2', 'W2', 'b2', 'b2', 'b2', 'W3', 'W3', 'W3', 'b3']
    assert(len(flatten_dict_of_matrices(wb_params)) == wb_p_values.shape[0])
    n_p = wb_p_values.shape[0]; epsilon = 1e-7  # print(n_p) # 42
    costs_plus, costs_minus, gradapprox = np.zeros((n_p, 1)), np.zeros((n_p, 1)), np.zeros((n_p, 1)) # init by zeros
    for i in range(n_p):  # if n_p=2 => i = 0, 1
        thetaplus =  np.copy(wb_p_values) # (47, 1)  # STEP 1
        thetaplus[i][0] = thetaplus[i][0] + epsilon  #;print(thetaplus[i][0])         # -0.3224171040135075 -0.38405425466841564 1.1337695423354375  ...
        #wbaz_cache = self.__forward(X, vector_to_dictionary(thetaplus, wb_params), L)
        wbaz_cache = forward_fun(X, vector_to_dictionary(thetaplus, wb_params), L)
        #costs_plus[i] = self.__compute_cost(wbaz_cache['A'+str(L)], Y, num_i, m, wbaz_cache, L2_lambd, L, DO_keep_prob, grad_check=True)
        costs_plus[i] = compute_cost_fun(wbaz_cache['A'+str(L)], Y, num_i, m, wbaz_cache, L2_lambd, L, DO_keep_prob, grad_check=True)
        thetaminus = np.copy(wb_p_values) # (47, 1)  # STEP 2                                     
        thetaminus[i][0] = thetaminus[i][0] - epsilon 
        wbaz_cache = forward_fun(X, vector_to_dictionary(thetaminus, wb_params), L)
        costs_minus[i] = compute_cost_fun(wbaz_cache['A'+str(L)], Y, num_i, m, wbaz_cache, L2_lambd, L, DO_keep_prob, grad_check=True)
        gradapprox[i] = (costs_plus[i] - costs_minus[i]) / (2 * epsilon) # STEP 3
    grad = gradients_to_vector(gradients) # backprop
    # find error value
    numerator = np.linalg.norm(grad - gradapprox)                                     # Step 1' # Compare gradapprox to backward propagation gradients by computing difference.
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                   # Step 2'
    difference = numerator / denominator                                              # Step 3'
    if difference > 2e-7: print ('iteration=',num_i,"\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else: print ('iteration=',num_i,"\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    return difference #             if error = 10^-7 -> great!   10^-5 -> maybe;   10^-3 -> worry
#if self.g_checking and i % (self.num_epochs/5) == 0: gradient_checking(self.X, self.Y, wb_parameters, grads, i, self._L, self._m, self.L2_lambd, self.DO_keep_prob, self.__forward, self.__compute_cost)







def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):  # Powers of two are often: 16, 32, 64, 128
    import numpy as np
    import math
    # X= (12288 rows-pixels, 148-examples) rand
    # Y= (1     row-class,   148-examples) bool
    np.random.seed(seed)
    m = X.shape[1] # 148
    mini_batches = []
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))  # [24, 7, 37 ... 148 elements from 1 to 148] # np.random.permutation(4) => # array([0, 3, 2, 1])
    shuffled_X = X[:, permutation]                # (12288, 148) # select all pixels, select all cols but using new order from permutation list 
    shuffled_Y = Y[:, permutation] # (1, 148) # .reshape((1,m)) is rudiment
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # 148/64 = 2.3125‬ =floor= 2 # math.floor(2.6)=2 # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches): # for k in range(0, 2): # k =  0, 1
        # get all rows-pixels, cols-examples: k=0 -> [0*64 : 1*64]; k=1 -> [1*64 : 2*64]
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size] # (12288, 64)
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size] # (1, 64)
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0: # 148%64 = 20
        # get all rows-pixels, cols-examples: [2*64 : end]
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:] # (12288, 20)
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:] # (1, 20)
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches # mini_batches[0][0].shape = (12288, 64), mini_batches[0][1].shape = (1, 64)  # len(mini_batches) = 3, len(mini_batches[0]) = 2 (tuple), 

# mini_batches = random_mini_batches(x_train, y_train , mini_batch_size=4) 
#for i in range(len(mini_batches)): print('\n batch 1 X shape = ', mini_batches[i][0].shape, '\n batch 1 Y shape = ', mini_batches[i][1].shape)
# Shuffle by permutation example:
    # X = np.random.randn(2,3)
        # array([[ 0.72437468, -2.04038084, -1.0797781 ],
        #        [-0.69342441, -2.33804615,  1.66226234]])
    # permutation = list(np.random.permutation(3)) # [1, 2, 0]
    # shuffled_X = X[:, permutation]
        # array([[-2.04038084, -1.0797781 ,  0.72437468],
        #        [-2.33804615,  1.66226234, -0.69342441]])



def generate_random_circles():
    # # EXAMPLE FROM c2w1a2
    # import h5py
    # import sklearn
    # import sklearn.datasets
    # import sklearn.linear_model
    # import scipy.io
    # def load_2D_dataset(plot=False):
    #     data = scipy.io.loadmat('datasets/data.mat')
    #     train_X = data['X'].T
    #     train_Y = data['y'].T
    #     test_X = data['Xval'].T
    #     test_Y = data['yval'].T
    #     if plot: plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.ravel(), s=40, cmap=plt.cm.Spectral);
    #     return train_X, train_Y, test_X, test_Y
    # x_train_ng, y_train_ng, x_test_ng, y_test_ng = load_2D_dataset(plot=True) # AndrewNG #x_train, y_train, x_test, y_test = load_2D_dataset(plot=True) # AndrewNG

    #x_train, y_train, x_test, y_test = load_2D_dataset() # AndrewNG #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    #model = my_l_layers_ffn( DO_keep_prob=1, layers_dims=[x_train.shape[0],20,3,1], num_iterations = 30000, learning_rate = 0.3, L2_lambd=0, X=x_train, Y=y_train, print_cost = True,  activation = 'relu', activation_l = 'sigmoid', loss='log')
    return

# ----------------------------------------------------------------------------------
# 1) NO OOP L HIDDEN LAYERS WITHOUT L2 and Dropout
    # #x_non_cols = ['fruit_label'] # uncomment this to plot 3d and use THREE FEATURES => COLOR_SCORE AND MASS AND width
    # x_non_cols = ['fruit_label','width'] # uncomment this to plot 2d and use TWO FEATURES => COLOR_SCORE AND MASS
    # #x_non_cols = ['fruit_label','width', 'mass'] # uncomment this to plot 1d and use ONLY ONE FEATURE => COLOR_SCORE
    # x_train, x_test, x_val = df_to_array(train,x_non_cols), df_to_array(test,x_non_cols), df_to_array(val,x_non_cols) # print('x_train',x_train.shape,'; y_train =',y_train.shape,'; x_test',x_test.shape,'; y_test',y_test.shape,'; x_val',x_val.shape,'; y_val',y_val.shape)

    # num_iterations = 20000
    # learning_rate = 0.0003
    # X=x_train
    # Y=y_train
    # # private
    # m = X.shape[1]
    # parameters = {}  
    # np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    # # new after one hidden layer
    # layers_dims = layers_dims=[x_train.shape[0],x_train.shape[0]*10,10,1] # [x_train.shape[0],2,1] # [3,2,1] # including input and output layers
    # activation = 'relu'      # hidden layers activation: relu or tanh
    # activation_l = 'sigmoid' # output layers activation: sigmoid usually
    # loss='log'               # cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    # L= len(layers_dims)-1    # number of layers (without input layer, but with output layer)

    # def __linear_forward(A_prev, W, b): return np.dot(W, A_prev) + b # W.dot(A_prev) + b
    # def __linear_activation_forward(Z, activation):
    #     if activation == "sigmoid": A = 1/(1+np.exp(-Z))
    #     elif activation == "sigmoid_fast": A = 1/(1+np.exp(-Z))
    #     elif activation == "relu":  A = np.maximum(0,Z)
    #     elif activation == "tanh":  A = np.tanh(Z)
    #     return A
    # def loss_backward(Y, AL, method):
    #     if method == "log": return -Y/AL + (1-Y)/(1-AL)
    #     if method == "log_sig_fast": return AL - Y # log + sigmoid can be computed faster: AL - Y
    # def linear_activation_backward(A, activation, dA=None, Z=None): # we know dA, we want find dAdZ (earlier A = g(Z)): 
    #     if activation == "relu": dAdZ = np.array(dA, copy=True); dAdZ[Z<=0] = 0; dAdZ[Z>0] = 1; return dAdZ
    #     elif activation == "sigmoid": return A*(1-A)
    #     elif activation == "sigmoid_fast": return 1
    #     elif activation == "tanh": return 1-A*A
    # def weights_backward(W, dZ, A_prev, method='linear'):
    #     m = A_prev.shape[1] # 12
    #     if method == "linear": # # Z2 = W2*A1 + b2        dZ2db2 = 1     dZ2dA1 = W2
    #         dW = np.dot(dZ, A_prev.T)/m                     # dW2=dZ2*dZ2dW2    dZ2dW2 = A1
    #         db = np.sum(dZ, axis = 1, keepdims = True)/m    # db2=dZ2*dZ2db2    dZ2db2 = 1
    #         dA_prev = np.dot(W.T, dZ)                       # dA1=dZ2*dZ2dA1    dZ2dA1 = W2
    #     return dW, db, dA_prev

    # def fit_L_hidden_layer():
    #     for l in range(1, L+1):
    #         parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01 / np.sqrt(layers_dims[l-1])
    #         parameters['b' + str(l)] = np.zeros((layers_dims[l], 1)) #pp.pprint(parameters) # dims[3,2,1] =>  {'W1':array([[ 0.9,-0.3,-0.3],[-0.6,0.4,-1.3]]), 'b1':array([[0.],[0.]]), 'W2':array([[1.2,-0.5]]), 'b2':array([[0.]])}    
    #     for i in range(num_iterations):
    #         # 1) FWD
    #         wbaz_cache = {'A0':X} # one_iter_cache
    #         A = X
    #         for li in range(1, L+1): # if L=2 => 1, 2
    #             l = str(li)
    #             A_prev = A
    #             W = parameters['W'+l]
    #             b = parameters['b'+l]
    #             act = activation_l if li==L else activation # act = 'tanh' if li<L else 'sigmoid'
    #             Z = __linear_forward(A_prev, W, b)
    #             A = __linear_activation_forward(Z, activation=act)
    #             wbaz_cache.update( {'A'+l:A, 'W'+l:W, 'b'+l:b, 'Z'+l:Z} )
    #         # 2) COST
    #         AL = wbaz_cache['A'+str(L)]    
    #         cost = (-1/m) * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL)) #;print('\nmean_cost=',cost); print('\ncosts=',(Y*np.log(A2) + (1-Y)*np.log(1-A2)).shape,Y*np.log(A2) + (1-Y)*np.log(1-A2))
    #         if i % 2000 == 0: print ("Cost %i: %f" %(i, cost))
    #         # 3) BCKWD
    #         c = wbaz_cache
    #         dA_prev = loss_backward(Y, c['A'+str(L)], loss) # same as dAL
    #         grads = {} # {'dA'+L:dA_prev}#
    #         for li in reversed(range(1,L+1)): # if L+1=3 => l=2,1
    #             l = str(li)
    #             act = activation_l if l==str(L) else activation
    #             dAdZ = linear_activation_backward(c['A'+l], act, dA_prev, c['Z'+l]) #if l==L: dZ = c['A'+l] - Y # alternative same speed  way (needs 'dZ'+l:dZ in grads) #else: dZ = np.dot(c['W'+str(li+1)].T,grads['dZ'+str(li+1)]) * (1 - np.power(c['A'+l], 2))
    #             dZ = dA_prev * dAdZ # dZ = dA_prev * linear_activation_backward(c['A'+l], act, dA_prev, c['Z'+l]) #if l==L: dZ = c['A'+l] - Y # alternative same speed  way (needs 'dZ'+l:dZ in grads) #else: dZ = np.dot(c['W'+str(li+1)].T,grads['dZ'+str(li+1)]) * (1 - np.power(c['A'+l], 2))
    #             dW, db, dA_prev = weights_backward(c['W'+l], dZ, c['A'+str(li-1)], 'linear')
    #             grads.update( {'dW'+l:dW, 'db'+l:db} ) # , 'dA'+str(li-1):dA_prev, 'dZ'+l:dZ, 'dA'+l+'dZ'+l:dAdZ
    #         # 4) UPDATE
    #         for l in range(0,L): # if L=2, li=0,1
    #             parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
    #             parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    #     return    
    # fit_L_hidden_layer()
# ----------------------------------------------------------------------------------








