import random as rn
import numpy as np
from mylib_model import my_l_layers_ffn
from mylib_core import random_mini_batches, gradient_checking, speed

class TrainMYNN():
    model = None
    conf = None
    history_callback = None

    def __init__(self, conf):
        self.conf = conf
        print('mynn init done')
        return
        
    def prepare_data(self, x_train, y_train, x_val, y_val): # norm np.arrays
        return

    def compile(self, x_train_shape):
        print('mynn compile done')
        return

    def fit(self, x_train, y_train, x_val, y_val):
        c = self.conf
        mini_batches = random_mini_batches(x_train.T, y_train.T, mini_batch_size=c['bs']) #for i in range(len(mini_batches)): print('\n batch 1 X shape = ', mini_batches[i][0].shape, '\n batch 1 Y shape = ', mini_batches[i][1].shape)


        if c['layers_dims_1']!=0: dims = [x_train.shape[1],x_train.shape[1]*c['layers_dims_1'],y_train.shape[1]]
        if c['layers_dims_2']!=0: dims = [x_train.shape[1],x_train.shape[1]*c['layers_dims_1'],x_train.shape[1]*c['layers_dims_2'],y_train.shape[1]],
        if c['layers_dims_3']!=0: dims = [x_train.shape[1],x_train.shape[1]*c['layers_dims_1'],x_train.shape[1]*c['layers_dims_2'],x_train.shape[1]*c['layers_dims_3'],y_train.shape[1]],
        if c['layers_dims_4']!=0: dims = [x_train.shape[1],x_train.shape[1]*c['layers_dims_1'],x_train.shape[1]*c['layers_dims_2'],x_train.shape[1]*c['layers_dims_3'],x_train.shape[1]*c['layers_dims_4'],y_train.shape[1]],
        if c['layers_dims_5']!=0: dims = [x_train.shape[1],x_train.shape[1]*c['layers_dims_1'],x_train.shape[1]*c['layers_dims_2'],x_train.shape[1]*c['layers_dims_3'],x_train.shape[1]*c['layers_dims_4'],x_train.shape[1]*c['layers_dims_5'],y_train.shape[1]],

        if c['layers_dims_2']!=0: dims = [item for sublist in dims for item in sublist] # flatten list of list, vs error: ([880, 880, 880, 880, 880, 880, 1],)

        #l_dims=[x_train.shape[1],x_train.shape[1]*c['layers_dims'][0],y_train.shape[1]],
        #if len(c['layers_dims'])==2: l_dims=[x_train.shape[1],x_train.shape[1]*c['layers_dims'][0],x_train.shape[1]*c['layers_dims'][1],y_train.shape[1]],
        #elif len(c['layers_dims'])==3: l_dims=[x_train.shape[1],x_train.shape[1]*c['layers_dims'][0],x_train.shape[1]*c['layers_dims'][1],x_train.shape[1]*c['layers_dims'][2],y_train.shape[1]],
        #dims = [x_train.shape[1],x_train.shape[1]*c['layers_dims'][0],y_train.shape[1]]
        #if len(c['layers_dims'])==2:   dims=[x_train.shape[1],x_train.shape[1]*c['layers_dims'][0],x_train.shape[1]*c['layers_dims'][1],y_train.shape[1]],
        #elif len(c['layers_dims'])==3: dims=[x_train.shape[1],x_train.shape[1]*c['layers_dims'][0],x_train.shape[1]*c['layers_dims'][1],x_train.shape[1]*c['layers_dims'][2],y_train.shape[1]],
        print('dims=',type(dims),len(dims),dims,list(dims))
        self.model = my_l_layers_ffn(
            optimizer=c['optimizer'], # adam # momentum, gd
            DO_keep_prob=c['DO_keep_prob'], #  # 1 turns off Dropout (0.86 - AndrewNG)
            L2_lambd=c['L2_lambd'], # 0 will turn off, 0.7 - changes from 90 degree plane to 20, 2.8 good
            layers_dims=dims,
            num_epochs = c['num_epochs'], # 20000
            learning_rate = c['learning_rate'], # 0.003 
            X=x_train.T, 
            Y=y_train.T, 
            mini_batches=mini_batches, 
            print_cost = bool(c['print_cost']),  
            activation = c['activation'], 
            activation_l = c['activation_l'], 
            loss=c['loss'], 
            lr_decay_type=c['lr_decay_type'], # no, exp 0.95 or 0.01; pop 0.01
            lr_decay_rate=c['lr_decay_rate'],  #  pop: more decay_rate, smaller learning rate (now can increase learning_rate), exp: more decay more lrate
            wb_init_type=c['wb_init_type'], 
            wb_init_coef=c['wb_init_coef'], # [0.9,0.09 for gd, 0.009 for adam] if all predictions == 1 => increse init_coef, if all==0 -> decrease
            g_checking=bool(c['g_checking']),
        )
        self.model.fit()
        self.history_callback = {'train_loss':self.model._costs}
        return

    def predict(self, x, df=None):
        #print('lgb predict done')
        return self.model.predict(x.T)
    
    def get_costs(self):
        #oy_train_list = self.history_callback.history["loss"]
        #oy_val_list = self.history_callback.history["val_loss"]
        #ox_list = [x for x in range(1,self.conf['epochs']+1)] # early stopping can cause error
        
        oy_train_list = self.history_callback["train_loss"]
        oy_val_list = oy_train_list
        ox_list = [x for x in range(1,len(oy_train_list)+1)]

        title = self.conf['name'] + '_'
        for t in self.conf['tune'].split(','): 
            if t!='': title += t+'_'+str(self.conf[t])+'_'
        return {'ox_list':ox_list, 'oy_train_list':oy_train_list, 'oy_val_list':oy_val_list, 'title':title}
