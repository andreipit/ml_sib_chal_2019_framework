import numpy as np
class my_l_layers_ffn: 
    # only init & public functions use global vars
    X = None
    Y = None
    #num_iterations = 200000
    num_epochs = 2000
    mini_batches = [('X1','Y1'),('X2','Y2'),('X3','Y3')]
    learning_rate = 0.0001
    print_cost = False
    layers_dims = [3,2,1] # including input and output layers # Andrew c2w1a2 -> [X.shape[0], 20, 3, 1]
    activation = 'relu' # hidden layers activation: relu or tanh
    activation_l = 'sigmoid' # output layers activation: sigmoid usually
    loss='log' # cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    L2_lambd = 0 # 0 will turn off, 0.7 - changes from 90 degree plane to 20
    DO_keep_prob = 1 # 1 turns off Dropout (0.86 - AndrewNG)
    optimizer = 'adam' # momentum, gd
    lr_decay_type='no' #'exp' 0.95 # 'pop', 0.001 good for 3d
    lr_decay_rate=0.95 # 0.001 for pop      pop: more decay_rate, smaller learning rate, exp: more decay more lrate
    wb_init_type='he'
    wb_init_coef=0.1 # if all predictions == 1 => increse init_coef, if all==0 -> decrease
    g_checking=False # comment this to see good checking -> (self.X, self.Y) = mini_batch
    _L= 2  # number of layers (without input layer, but with output layer)
    _m = 12
    _costs = []
    _parameters = {}
    
    def __init__(self, layers_dims=[3,2,1],  num_epochs=2000, learning_rate=0.0001, L2_lambd=0, DO_keep_prob=1, X=np.array([1,2,3]), Y=np.array([4,5,6]), print_cost=False,  activation='relu', activation_l='sigmoid', loss='log', mini_batches=None, optimizer='gd', lr_decay_type='exp', lr_decay_rate='0.95', wb_init_type='he', wb_init_coef=0.01, g_checking=False):
        self.X = X; self.Y = Y; # oldway: num_iterations=20000, #self.num_iterations = num_iterations;
        self.num_epochs = num_epochs; self.learning_rate = learning_rate; self.print_cost = print_cost; self.layers_dims = layers_dims; self.activation = activation; self.activation_l = activation_l; self.loss=loss; self.L2_lambd=L2_lambd; self.DO_keep_prob = DO_keep_prob; self.mini_batches=mini_batches; self.optimizer=optimizer; self.lr_decay_type=lr_decay_type; self.lr_decay_rate=lr_decay_rate; self.wb_init_type=wb_init_type; self.wb_init_coef=wb_init_coef; self.g_checking=g_checking
        self._L = len(layers_dims) - 1 # without input layer
        self._m = X.shape[1] # 12 examples
        self._costs=[]; self._parameters={}; # usefull, when creating second object of this class
    
    def __learning_rate_decay(self, learning_rate,  epoch_num, decay_type='no', decay_rate=0.95):
        def nozero(x): return 1e-8 if x==0 else x
        if decay_type=='pop': r = learning_rate / nozero(1 + decay_rate * epoch_num) #0.001 learning rate decay: alpha = alpha_0 / [1 + decay_rate * epoch_num]
        elif decay_type=='exp': r = decay_rate * epoch_num * learning_rate # alpha = 0.95^epoch_num * alpha_0
        elif decay_type=='discrete': r = decay_rate * learning_rate / nozero(np.sqrt(epoch_num)) # not working
        elif decay_type=='no': r = learning_rate
        return r
    
    def __init_parameters(self,layers_dims,method='he_small',init_coef=0.01,optimizer='gd'): # init_coef=1 will turn of, result in range [-1;1]
        np.random.seed(2)
        # 1) init weights
        parameters = {}
        for l in range(1, len(layers_dims)): # if layers_dims=[x_train.shape[0],10,1] layers_dims=3 => l=1,2
            ran = np.random.randn(layers_dims[l], layers_dims[l-1])
            if   method == "he":          W = ran * init_coef * np.sqrt(2./layers_dims[l-1])    # 91/91  PROS: BEST!!!, recommended method for relu [0.01, 0.1, 1]
            elif method == "rand":        W = ran * init_coef                                   # 
            elif method == "xavier":      W = ran * init_coef * np.sqrt(1./layers_dims[l-1])    # 83/83  CONS: bad for relu, better for tanh then he
            elif method == "my_old":      W = ran * 0.01 / np.sqrt(layers_dims[l-1])       # 83/83  was my default (maybe Ive made error)
            elif method == "zero":        W = np.zeros((layers_dims[l], layers_dims[l-1])) # 50/50  CONS: fails to break symmetry
            parameters['W' + str(l)] = W
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1)) #print('l=',l,'layers_dims[l-1]=',layers_dims[l-1],' np.sqrt(2./layers_dims[l-1])=',np.sqrt(2./layers_dims[l-1]))
        # 2) init optimizer
        v = {}; s={}; t=0
        for l in range(len(layers_dims)-1):
            if optimizer == "gd": pass # no initialization required for gradient descent
            elif optimizer == "momentum": #v = initialize_velocity(parameters)
                v["dW" + str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
                v["db" + str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
            elif optimizer == "adam":
                v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
                v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
                s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
                s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        opt_parameters = {'v':v, 's':s, 't':t}
        return parameters, opt_parameters # dims[3,2,1] =>  {'W1':array([[ 0.9,-0.3,-0.3],[-0.6,0.4,-1.3]]), 'b1':array([[0.],[0.]]), 'W2':array([[1.2,-0.5]]), 'b2':array([[0.]])}    
    
    def __forward(self, X, p, L, DO_keep_prob=1):
        def nozeroarr(x): x[x==0]=1e-8; return x
        def nozero(x): return 1e-8 if x==0 else x
        def __linear_forward(A_prev, W, b): return np.dot(W, A_prev) + b # W.dot(A_prev) + b
        def __linear_activation_forward(Z, activation):
            if activation == "sigmoid": A = 1/nozeroarr(1+np.exp(-Z)) # A = 1/(1+expit(-Z))  # vs RuntimeWarning: overflow encountered in exp
            elif activation == "sigmoid_fast":  A = 1/nozeroarr(1+np.exp(-Z)) # A = 1/(1+expit(-Z))  
            elif activation == "relu":  A = np.maximum(0,Z)
            elif activation == "tanh":  A = np.tanh(Z)
            elif activation == "softmax": Z_exp = np.exp(Z); Z_sum = np.sum(Z_exp, axis = 0, keepdims = True); A =  Z_exp / nozeroarr(Z_sum) # A = np.exp(Z)/np.sum(np.exp(Z), axis = 0, keepdims = True)
            return A
        def add_dropout(A, DO_keep_prob): # disable score and predict before debug
            D = np.random.rand(A.shape[0], A.shape[1]) #;print('A v1',A,'\nD v1',D) # Step 1: initialize matrix D1 = np.random.rand(..., ...)
            D = D <= DO_keep_prob#    ; print('\nD v2',D) # earlier was only '<'    # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
            A = A * D      #;print('\nA v2',A)                                      # Step 3: shut down some neurons of A1
            A = A / nozero(DO_keep_prob) #;print('\nA v3',A)                                # Step 4: scale the value of neurons that haven't been shut down
            return A,D
        one_iter_cache = {'A0':X}
        A = X
        for li in range(1, L+1): # if L=2 => 1, 2
            A_prev = A; l = str(li); act = self.activation_l if li==L else self.activation # act = 'tanh' if li<L else 'sigmoid'
            Z = __linear_forward(A_prev, p['W'+l], p['b'+l]) # Z = __linear_forward(A_prev, W, b)
            A = __linear_activation_forward(Z, activation=act)
            if DO_keep_prob!=1 and li!=L: A,D = add_dropout(A, DO_keep_prob); one_iter_cache.update( {'D'+l:D} ) # not apply to the input layer or output, work without cost plotting, use only in training
            one_iter_cache.update( {'A'+l:A, 'W'+l:p['W'+l], 'b'+l:p['b'+l], 'Z'+l:Z} ) # one_iter_cache.update( {'A'+l:A, 'W'+l:W, 'b'+l:b, 'Z'+l:Z} )
        #print('Z'+str(L),Z)
        #print('A'+str(L),A)
        #print('A'+str(L)+' col 0',A[:,0],'\n first A col sum=',np.sum(A[:,0])) # 1
        return one_iter_cache

    def __compute_cost(self, AL, Y, i, m, c, L2_lambd, L, DO_keep_prob, grad_check=False):
        #e = 0.00000001 # Epsila - vs error 'divide by zero encountered in log'
        step = 300; w_sum=0; #logprobs = np.multiply(np.log(AL),Y) + np.multiply(np.log(1-AL),1-Y) #cost = - np.sum(logprobs) / m
        #logprobs = np.multiply(-np.log(AL),Y) + np.multiply(-np.log(1 - AL), 1 - Y); cost = 1./m * np.nansum(logprobs)   # Andrew NG version (c2w1a2 regutils)
        if self.activation_l=='sigmoid': #cost = (-1/m) * np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL)) #;print('\nmean_cost=',cost); print('\ncosts=',(Y*np.log(A2) + (1-Y)*np.log(1-A2)).shape,Y*np.log(A2) + (1-Y)*np.log(1-A2))
            costs_all = Y*np.log(AL) + (1-Y)*np.log(1-AL); cost = (-1/m) * np.sum(costs_all) #print('\nY=',Y.shape,'\n',Y,'\nAL=',AL.shape,'\n',AL,'\nnp.log(AL)=',np.log(AL).shape,'\n',np.log(AL),'\ncosts_all=',costs_all.shape,'\n',costs_all,'\ncost=',cost.shape,'\n',cost)
        elif self.activation_l=='softmax': 
            J = Y * np.log(AL); 
            costs_all = np.sum(J, axis=0) # removes zeros, summing all values in each col
            cost = (-1/m)*np.sum(costs_all) # print('\nY=',Y.shape,'\n',Y,'\nAL=',AL.shape,'\n',AL,'\nnp.log(AL)=',np.log(AL).shape,'\n',np.log(AL),'\nJ=',J.shape,'\n',J,'\ncosts_all=',costs_all.shape,'\n',costs_all,'\ncost=',cost.shape,'\n',cost)
        if L2_lambd!=0: # when lambd==0 => Cost after iteration 18000: 0.471168, when 0.7 => 0.485028
            for l in range(1,L+1): w_sum += np.sum(np.square(c['W'+str(l)])) # if L=2, l=1,2   #L2_regularization_cost = L2_lambd * (np.sum(np.square(c['W1'])) + np.sum(np.square(c['W2']))) / (2 * m)
            cost = cost + L2_lambd * w_sum / (2 * m)
        if grad_check==False: # grad_check doesnt work with dropout, but works with L2
            if DO_keep_prob != 1: cost = 0 # 'cost is not correct, when using Dropout. Instruction: set all keep_prob=1, plot cost till you get nearly correct graph, set your keep_probs and work without cost plotting
            if i % step == 0: self._costs.append(cost)
            if self.print_cost and i % step == 0: print ("Cost after iteration %i: %f" %(i, cost))
        return cost
    
    def __backward(self, Y, c, L, m, L2_lambd, DO_keep_prob):
        def nozeroarr(x): x[x==0]=1e-8; return x
        def nozero(x): return 1e-8 if x==0 else x
        grads = {} # {'dA'+L:dA_prev}#   #cost = (-1/m) * np.sum( Y*np.log(AL)+(1-Y)*np.log(1-AL) )  +  L2_lambd * (np.sum(np.square(c['W1'])) + np.sum(np.square(c['W2']))) / (2 * m)
        def loss_backward(Y, AL, method, activation):
            if method == "log": 
                if activation == "sigmoid":   return -Y/nozeroarr(AL) + (1-Y)/nozeroarr(1-AL) #if method == "log_sig_fast": return AL - Y # log + sigmoid can be computed faster: dZL = AL - Y
                elif activation == "softmax": return -Y/nozeroarr(AL) + (1-Y)/nozeroarr(1-AL) # for softmax retun must be -Y/AL, but we use sigmoid version to get right derivative, cause I dont know dAdZ for softmax
        def linear_activation_backward(A, activation, dA=None, Z=None): # we know dA, we want find dAdZ (earlier A = g(Z)): 
            if activation == "relu_AndrewNG": np.int64(A > 0) # Andrew NG version from c2w1a2 reg_utils
            elif activation == "relu": dAdZ = np.array(dA, copy=True); dAdZ[Z<=0] = 0; dAdZ[Z>0] = 1; return dAdZ
            elif activation == "sigmoid": return A*(1-A) # elif activation == "sigmoid_fast": return 1
            elif activation == "softmax": return A*(1-A) #  for softmax I dont know, but we use sigmoid version to get right derivative
            elif activation == "tanh": return 1-A*A
        def weights_backward(W, dZ, A_prev, method='linear'):
            m = A_prev.shape[1] # 12
            if method == "linear": # # Z2 = W2*A1 + b2        dZ2db2 = 1     dZ2dA1 = W2
                dW = np.dot(dZ, A_prev.T)/m                     # dW2=dZ2*dZ2dW2    dZ2dW2 = A1
                db = np.sum(dZ, axis = 1, keepdims = True)/m    # db2=dZ2*dZ2db2    dZ2db2 = 1
                dA_prev = np.dot(W.T, dZ)                       # dA1=dZ2*dZ2dA1    dZ2dA1 = W2
            return dW, db, dA_prev
        def add_dropout(dA, D, DO_keep_prob): # disable score and predict before debug
            dA = dA * D      #;print('\nA v2',A)                                      # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
            dA = dA / nozero(DO_keep_prob) #;print('\nA v3',A)                                # Step 2: Scale the value of neurons that haven't been shut down
            return dA
        dA_prev = loss_backward(Y, c['A'+str(L)], self.loss, self.activation_l) # same as dAL, L2_cost = 0, cause depends only from weights
        for li in reversed(range(1,L+1)): # if L+1=3 => li=2,1
            l = str(li); act = self.activation_l if li==L else self.activation
            if DO_keep_prob!=1 and li!=L: dA_prev = add_dropout(dA_prev, c['D'+l],DO_keep_prob); # not apply to the input layer or output, work without cost plotting, use only in training
            dAdZ = linear_activation_backward(c['A'+l], act, dA_prev, c['Z'+l]) #if l==L: dZ = c['A'+l] - Y # alternative same speed  way (needs 'dZ'+l:dZ in grads) #else: dZ = np.dot(c['W'+str(li+1)].T,grads['dZ'+str(li+1)]) * (1 - np.power(c['A'+l], 2))
            dZ = dA_prev * dAdZ # dZ = dA_prev * linear_activation_backward(c['A'+l], act, dA_prev, c['Z'+l]) #if l==L: dZ = c['A'+l] - Y # alternative same speed  way (needs 'dZ'+l:dZ in grads) #else: dZ = np.dot(c['W'+str(li+1)].T,grads['dZ'+str(li+1)]) * (1 - np.power(c['A'+l], 2))
            dW, db, dA_prev = weights_backward(c['W'+l], dZ, c['A'+str(li-1)], 'linear')
            if L2_lambd!=0: dW += L2_lambd*c['W'+l]/m # L2 changes only this !!!
            grads.update( {'dW'+l:dW, 'db'+l:db} ) # , 'dA'+str(li-1):dA_prev, 'dZ'+l:dZ, 'dA'+l+'dZ'+l:dAdZ
        return grads
    
    def __update_parameters(self, parameters, opt_parameters, grads, L, learning_rate, optimizer):
        def nozeroarr(x): x[x==0]=1e-8; return x
        def nozero(x): return 1e-8 if x==0 else x
        #L = len(parameters) // 2 # 2 = number of layers in the neural network
        beta = 0.9; beta1 = 0.9; beta2 = 0.999; epsilon = 1e-8; v = opt_parameters['v']; s = opt_parameters['s']; t = opt_parameters['t'];
        v_corrected = {}                         # Initializing first moment estimate, python dictionary
        s_corrected = {}                         # Initializing second moment estimate, python dictionary
        for l in range(L):
            if optimizer == "gd":
                parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
                parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
            elif optimizer == "momentum": #parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
                v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1 - beta) * grads["dW" + str(l+1)]
                v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1 - beta) * grads["db" + str(l+1)]
                parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
                parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]
            elif optimizer == "adam":
                t = t + 1 # Adam counter #parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,  epsilon)
                # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
                v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
                v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]
                # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
                v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / nozero(1 - beta1)
                v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / nozero(1 - beta1)
                # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
                s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * (grads["dW" + str(l+1)] * grads["dW" + str(l+1)])
                s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * (grads["db" + str(l+1)] * grads["db" + str(l+1)])
                # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
                s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / nozero(1 - beta2)
                s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / nozero(1 - beta2)
                # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
                parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / nozeroarr(np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
                parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / nozeroarr(np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
        return parameters
    
    def score(self, X=None, Y=None): # default - train accuracy
        if X is None: X=self.X; Y=self.Y
        Y_pred = self.predict(X) #default=y_train_pred , DO_keep_prob=self.DO_keep_prob
        if    self.activation_l=='sigmoid': errors = np.abs(Y_pred - Y) #train_accuracy = 100 - np.mean(np.abs(y_train_pred - self.Y)) * 100
        elif  self.activation_l=='softmax': 
            Y = np.argmax(Y, axis=0); errors = np.zeros(len(Y));
            for i in range(len(Y)): errors[i] = 1 if Y[i] != Y_pred[i] else 0
        return 100 - np.mean(errors * 100)

    def predict(self, X): # , DO_keep_prob=1
        m = X.shape[1]
        p = np.zeros((1,m))
        wbaz_cache = self.__forward(X, self._parameters, self._L, DO_keep_prob=1) # A common MISTAKE !!!' when using dropout is to use it both in training and testing. You should use dropout (randomly eliminate nodes) only in training
        AL = wbaz_cache['A'+str(self._L)]
        if    self.activation_l=='softmax': p = np.argmax(AL, axis=0)#.tolist()
        elif  self.activation_l=='sigmoid':
            for i in range(0, m): p[0,i] = 1 if AL[0,i] > 0.5 else 0 # AL.shape[1]
        return p 
    
    def fit(self):
        wb_parameters, opt_parameters = self.__init_parameters(self.layers_dims, self.wb_init_type, self.wb_init_coef, self.optimizer) #; pp.pprint(wb_parameters) # if all predictions == 1 => increse init_coef
        for i in range(self.num_epochs):  # old name - num_iterations
            learning_rate_decay = self.__learning_rate_decay(self.learning_rate, i, self.lr_decay_type, self.lr_decay_rate) # 'exp', 0.95; 'pop', 0.001 good for 3d
            for mini_batch in self.mini_batches: #for j in range(self._m):  # stochastic GD
                (self.X, self.Y) = mini_batch if not self.g_checking else (self.X, self.Y)
                self._m = self.X.shape[1]
                wbaz_cache = self.__forward(self.X, wb_parameters, self._L, self.DO_keep_prob) #; pp.pprint(wbaz_cache);
                cost = self.__compute_cost(wbaz_cache['A'+str(self._L)], self.Y, i, self._m, wbaz_cache, self.L2_lambd, self._L, self.DO_keep_prob)
                grads = self.__backward(self.Y, wbaz_cache, self._L, self._m, self.L2_lambd, self.DO_keep_prob)                 # if i==self._L: pp.pprint(grads) # never do each loop!!!
                if self.g_checking and i % (self.num_epochs/5) == 0: gradient_checking(self.X, self.Y, wb_parameters, grads, i, self._L, self._m, self.L2_lambd, self.DO_keep_prob, self.__forward, self.__compute_cost)
                wb_parameters = self.__update_parameters(wb_parameters, opt_parameters, grads, self._L, learning_rate_decay, self.optimizer) # ; pp.pprint(wb_parameters)
        self._coef = {"costs" : self._costs, "learning_rate" : self.learning_rate, "num_epochs": self.num_epochs}
        self._parameters = wb_parameters                                                       # ; pp.pprint(wb_parameters)
        return
    