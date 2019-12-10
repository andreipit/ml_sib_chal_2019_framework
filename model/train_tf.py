import random as rn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class TrainTF():
    optimizer = None
    model = None
    history_callback = None
    conf = None

    def __init__(self, conf):
        self.conf = conf
        tf.random.set_seed(self.conf['rs'])

        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=50,#50 100000,
            decay_rate=0.96,
            staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        #tf.keras.backend.clear_session()
        #print('tf init done')
        return
    def prepare_data(self, x_train, y_train, x_val, y_val): # norm np.arrays
        return
    def compile(self, x_train_shape):
        data_in = keras.Input(shape=(x_train_shape[1],), name="img_in")
        dense_1 = layers.Dense(32, activation="relu")(data_in)
        data_out = layers.Dense(1, activation="linear")(dense_1)
        # dense_1 = layers.Dense(x_train_shape[1], activation="relu")(data_in)
        # dense_2 = layers.Dense(32, activation="relu")(dense_1)
        # data_out = layers.Dense(1, activation="linear")(dense_2)
        self.model = keras.Model(data_in, data_out)
        self.model.compile(optimizer=self.optimizer, loss='mape') #mse  mae optimizer='adam'
        #print('tf define done')
        return

    def fit(self, x_train, y_train, x_val, y_val):
        #tf.keras.backend.clear_session()
        self.history_callback = self.model.fit(x_train,y_train,
                  epochs=self.conf['epochs'],
                  batch_size=self.conf['bs'],
                  shuffle=False,
                  validation_data=(x_val, y_val),
                  #callbacks=[keras.callbacks.EarlyStopping(patience=5)],
                  verbose=0,
                 )
        #print('tf fit done')
        return

    def predict(self, x, df=None):
        #print('tf predict done')
        return self.model.predict(x)
        
    def get_costs(self):
        oy_train_list = self.history_callback.history["loss"]
        oy_val_list = self.history_callback.history["val_loss"]
        #ox_list = [x for x in range(1,self.conf['epochs']+1)] # early stopping can cause error
        ox_list = [x for x in range(1,len(oy_train_list)+1)]
        # title_unique = 'tf_epochs='+str(self.conf['epochs']) + '_bs=' + str(self.conf['bs']) + '_rs=' + str(self.conf['rs'])
        # #title = str(self.conf)
        # title = title_unique# + '_score_train='+str(round(self.conf['score_train'],2)) + '_score_val='+str(round(self.conf['score_val'],2))
        title = self.conf['name'] + '_'
        for t in self.conf['tune'].split(','): 
            if t!='': title += t+'_'+str(self.conf[t])+'_'


        
        return {'ox_list':ox_list, 'oy_train_list':oy_train_list, 'oy_val_list':oy_val_list, 'title':title}



    # def define(self):
    #     initial_learning_rate = 0.001
    #     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #         initial_learning_rate,
    #         decay_steps=50,#100000,
    #         decay_rate=0.96,
    #         staircase=True)
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    #     data_in = keras.Input(shape=(tr_data.shape[1]-1,), name="img_in")
    #     dense_1 = layers.Dense(32, activation="relu")(data_in)
    #     data_out = layers.Dense(1, activation="linear")(dense_1)
    #     model = keras.Model(data_in, data_out)
    #     model.compile(optimizer=optimizer, loss='mse') # mae optimizer='adam'
    #     return model

    # def train(self, x_train, y_train, params={'epochs':10, bs:'1024'}, x_val=None, y_val=None):
    #     model.fit(
    #         x_train, y_train,
    #         epochs=params['epochs'],
    #         batch_size=params['batch_size'],
    #         shuffle=False,
    #         validation_data=(x_val, y_val),
    #         callbacks=[keras.callbacks.EarlyStopping(patience=5)],
    #         verbose=0,
    #     )
    #     return model

    # def predict_model(self, model, x):
    #     return
    #     # tr_preds = model.predict(trn) * tr_data["activity"].std() + tr_data["activity"].mean()
    #     # cv_preds = model.predict(cvn) * tr_data["activity"].std() + tr_data["activity"].mean()

    #     # tr_preds = pd.Series(tr_preds.flatten(), index=tr_data.index, name="activity_pred").sort_index()
    #     # cv_preds = pd.Series(cv_preds.flatten(), index=cv_data.index, name="activity_pred")

    #     # def mape(y_true, y_pred): return np.mean(np.abs((y_pred-y_true)/y_true))
    #     # y_train_score = 100 * mape(tr_data['activity'].values, tr_preds.values):.2f
    #     # y_val_score = 100 * mape(cv_data['activity'].values, cv_preds.values):.2f
    #     # print(f"MAPE (training set): {y_train_score}%")
    #     # print(f"MAPE (cross-validation set): {y_val_score}%")
    #     # return y_train_score, y_val_score


