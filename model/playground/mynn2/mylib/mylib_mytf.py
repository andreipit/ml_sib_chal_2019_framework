# MY TF INTERFACE

class mytensorflow: 
    num_const = -1
    num_tensor_by_name = {}
    num_mul = -1
    num_sess = -1
    x = None
    int64 = 'int64'
    def __init__(self,x=0):
        self.x = x; num_tensor_by_name = {};
    def constant(self, x, name='noname'):
        temp = mytensorflow().python().framework().ops().Tensor(root=self,var_or_fun=x,mode='Const',name=name)
        return temp
    def multiply(self, x, y, name='noname'):
        def mult(x,y):
            return x.var_or_fun * y.var_or_fun
        #result = x.x * y.x
        return mytensorflow().python().framework().ops().Tensor(root=self,var_or_fun=mult,mode='Mul',name=name,arg1=x,arg2=y)
    def Variable(self, var_or_fun, name='noname', arg1=0, arg2=0):
        def saved_fun(x=None,y=None):
            return var_or_fun.var_or_fun
#         def mult(x,y):
#             return x.var_or_fun * y.var_or_fun
        fake_arg1 = mytensorflow().python().framework().ops().Tensor(root=self,var_or_fun=var_or_fun,mode='VarWrapper',name=name,arg1=0,arg2=0)
        return mytensorflow().python().framework().ops().variables.RefVariable(root=self,var_or_fun=saved_fun,mode='Var',name=name,arg1=fake_arg1,arg2=0)
    def global_variables_initializer(self):
        def initialize_smth(x,y):
            return x.var_or_fun - y.var_or_fun
        return mytensorflow().python().framework().ops().Operation(root=self,var_or_fun=initialize_smth)
    
    def placeholder(self, DType='tf.int64', name = 'x', x=1):
        return mytensorflow().python().framework().ops().Tensor(root=self,var_or_fun=x,mode='Placeholder',name=name)
    
    def reset_default_graph(self):
        self.num_const = -1
        self.num_mul = -1
    def Session(self):
        return mytensorflow().python().client().session().Session(self)
    class python:
        class client:
            class session:
                class Session:
                    x = None
                    root = None
                    count=0
                    _closed=True
                    graph=None # graph.version # if self.graph.version == 0:  raise RuntimeError('The Session graph is empty.  Add operations to the '
                    def __enter__(self): return self
                    def __exit__(self, exc_type, exc_val, exc_tb): return
                    def __init__(self, root, x=0):
                        self.x = x
                        self.root = root
                        root.num_sess += 1
                        self.count = root.num_sess
                        self._closed = False
                    def run(self, x,  feed_dict = {}):
                        if self._closed:
                            raise RuntimeError('Attempted to use a closed Session.')
                        if bool(feed_dict): # if not empty
                            return x * int(feed_dict[list(feed_dict.keys())[0]])
                        else: return x.var_or_fun(x.arg1, x.arg2)
                    def close(self):
                        self.root.num_sess = -1
                        self.count = -1
                        self._closed = True
                        return
        class framework:
            class ops: 
                class Operation:  # tensorflow.python.framework.ops.Operation
                    var_or_fun = None
                    arg1=0 
                    arg2=0
                    def __init__(self, root=None, var_or_fun=0, arg1=0, arg2=0):
                        self.var_or_fun = var_or_fun; 
                        self.arg1=mytensorflow().python().framework().ops().Tensor(root=root,var_or_fun=0,mode='Init',name='Init'); 
                        self.arg2=mytensorflow().python().framework().ops().Tensor(root=root,var_or_fun=0,mode='Init',name='Init'); 
                class Tensor: 
                    var_or_fun = None 
                    arg1=0 
                    arg2=0
                    name = 'noname'
                    root = None
                    mode='Const'
                    count_total=0
                    def __init__(self, root=None, var_or_fun=0, mode='Const', name='noname', arg1=0, arg2=0):
                        self.var_or_fun = var_or_fun; 
                        self.root = root; self.mode = mode; self.arg1 = arg1; self.arg2 = arg2
                        if mode=='Const': 
                            root.num_const += 1
                            self.count_total = root.num_const
                        elif mode=='Mul': 
                            root.num_mul += 1
                            self.count_total = root.num_mul
                        root.num_tensor_by_name[name] = 1 if name not in root.num_tensor_by_name else root.num_tensor_by_name[name]+1
                        self.name = name + '_' + str(root.num_tensor_by_name[name]) + ':0';     
                    def __sub__(x, y):
                        result = x.var_or_fun - y.var_or_fun
                        #x.var_or_fun = result
                        temp = mytensorflow().python().framework().ops().Tensor(root=x.root,var_or_fun=result,mode='Sub',name=x.name)
                        return temp
                    def __rmul__(x, y):
                        #result = x * y.var_or_fun
                        #y.var_or_fun = result
                        #temp = mytensorflow().python().framework().ops().Tensor(root=y.root,var_or_fun=result,mode='mult_overload',name=y.name)
                        return y * x.var_or_fun
                    def __repr__(self):
                        shape = ''
                        if type(self.var_or_fun) == type(0): shape='()'
                        return repr("<tf.Tensor '" + self.mode + "_" + str(self.count_total) + ":0' shape=" + shape + " dtype=" + str(type(self.var_or_fun).__name__) + ">") # <tf.Tensor 'Const_18:0' shape=() dtype=int32>
                class variables:  #tensorflow.python.ops.variables.RefVariable
                    class RefVariable:
                        var_or_fun = None; arg1=0; arg2=0;
                        name = 'noname'
                        root = None
                        mode='Const'
                        count_total=0
                        def __init__(self, root, var_or_fun=0, mode='Const', name='noname', arg1=0, arg2=0):
                            self.var_or_fun = var_or_fun; 
                            self.root = root; self.mode = mode; self.arg1 = arg1; self.arg2 = arg2
                            if mode=='Const': 
                                root.num_const += 1
                                self.count_total = root.num_const
                            elif mode=='Mul': 
                                root.num_mul += 1
                                self.count_total = root.num_mul
                            root.num_tensor_by_name[name] = 1 if name not in root.num_tensor_by_name else root.num_tensor_by_name[name]+1
                            self.name = name + '_' + str(root.num_tensor_by_name[name]) + ':0';     
                        def __repr__(self):
                            shape = ''
                            if type(self.var_or_fun) == type(0): shape='()'
                            return repr("<tf.Tensor '" + self.mode + "_" + str(self.count_total) + ":0' shape=" + shape + " dtype=" + str(type(self.var_or_fun).__name__) + ">") # <tf.Tensor 'Const_18:0' shape=() dtype=int32>
mytf = mytensorflow()    
mytf.reset_default_graph()
sess = mytf.Session()
# 1) constants -------------------------------
a = mytf.constant(2); b = mytf.constant(10); c = mytf.multiply(a,b)
with mytf.Session() as sess: print(sess.run(c)) # 20
# 2) variable and init -------------------------------
y_hat = mytf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = mytf.constant(49, name='y')                    # Define y. Set to 39
loss = mytf.Variable((y - y_hat), name='loss')
init = mytf.global_variables_initializer()         # When init is run later (session.run(init)),
with mytf.Session() as sess: sess.run(init); print(sess.run(loss)) # 13
# 3) placeholder -------------------------------
x = mytf.placeholder(mytf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 3})) # 6

# 20
# 13
# 6



# -------------------------------
# -------------------------------
# -------------------------------
# ORIGINAL TF

# 1) constants -------------------------------
#tf.reset_default_graph()
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
#print(type(c))
#print(a - b)
#print('b=',b)
sess = tf.Session()
print(sess.run(c))
sess.close()
# 2) variable and init -------------------------------
y_hat = tf.constant(36, name='y_hat')            # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')                    # Define y. Set to 39
loss = tf.Variable((y - y_hat)**2, name='loss')  # Create a variable for the loss
init = tf.global_variables_initializer()         # When init is run later (session.run(init)),
with tf.Session() as session:                    # Create a session and print the output
    session.run(init)                            # Initializes the variables
    print(session.run(loss))                     # Prints the loss
# 3) placeholder -------------------------------
# Change the value of x in the feed_dict
sess = tf.Session()
x = tf.placeholder(tf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 3}))
sess.close()    
# 20
# 9
# 6