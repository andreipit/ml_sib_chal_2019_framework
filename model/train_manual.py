import random as rn
import numpy as np

class TrainMANUAL():
    conf = None

    def __init__(self, conf):
        self.conf = conf
        return
        
    def prepare_data(self, x_train, y_train, x_val, y_val): # norm np.arrays
        return

    def compile(self, x_train_shape):
        return

    def fit(self, x_train, y_train, x_val, y_val):
        return

    def predict(self, x, df=None):
        start = '2018-04-04 00:05:00'
        end = '2018-04-05 05:05:00'
        def f(fnum): return df['f'+str(fnum)]#[0]#round(train_data['f'+str(fnum)][0],4)
        mean_weight = df[['f53','f54','f55']].max(axis=1) #/1000 + 30)


        # k1=0.3; k2=2; d2=1000; b=0 #score_train = 11.762662661642317 score_val = 10.571245
        # k1=0.1; k2=1; d2=400; b=0
        #k1=0.2; k2=2; d2=800; b=0 # score_train = 13.003379282112896 score_val = 11.973854874512089

        c = self.conf
        k1=c['k1']; k2=c['k2']; d2=c['d2']; b=c['b']

        formula = f(35)*k1 + mean_weight*k2 + b#/ ( (catal_r1/prop_r1 + catal_r2/prop_r2) )
        return np.array(formula/d2)
    
    def get_costs(self):
        ox_list=None
        oy_train_list=None
        oy_val_list=None
        title=None
        return {'ox_list':ox_list, 'oy_train_list':oy_train_list, 'oy_val_list':oy_val_list, 'title':title}

