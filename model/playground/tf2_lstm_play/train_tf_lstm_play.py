import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import utils_lstm as utils_lstm
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
tf.random.set_seed(13)

def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []; labels = []
    start_index = start_index + history_size
    if end_index is None: end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        if single_step: labels.append(target[i+target_size])
        else: labels.append(target[i:i+target_size])
    return np.array(data), np.array(labels)




class TrainTFLSTM():
    optimizer = None
    model = None
    history_callback = None
    conf = None
    train_data_multi = None
    val_data_multi = None
    input_shape = None
    
    def __init__(self, conf):
        self.conf = conf
        tf.random.set_seed(self.conf['rs'])
        return

    def prepare_data(self, x_train, y_train, x_val, y_val): # norm np.arrays
        c = self.conf
        STEP = c['step']
        BUFFER_SIZE = c['buffer_size']
        BATCH_SIZE = c['batch_size']
        past_history = c['past_history']
        future_target = c['future_target']
        TRAIN_SPLIT = x_train.shape[0]
        def joinback_train_val_x_y(x1, y1, x2, y2):
            train = np.concatenate((y1,x1), axis=1)
            val   = np.concatenate((y1,x1), axis=1)
            dataset = np.concatenate((train,val), axis=0)
            return dataset # big np array, like original train_data df
        dataset = joinback_train_val_x_y(x_train, y_train, x_val, y_val)
        x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0, TRAIN_SPLIT, past_history, future_target, STEP)
        x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0], TRAIN_SPLIT, None, past_history, future_target, STEP)
        

        print('x_train_multi',x_train_multi.shape, 'y_train_multi',y_train_multi.shape)
        #print('x_val_multi',x_val_multi.shape, 'y_val_multi',y_val_multi.shape)
        # join cols (tf input format)
        train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
        train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
        val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
        self.train_data_multi = train_data_multi
        self.val_data_multi = val_data_multi
        self.input_shape = x_train_multi.shape[-2:]
        return 
    
    def compile(self, x_train_shape):
#         data_in = keras.Input(shape=(x_train_shape[1],), name="img_in")
#         dense_1 = layers.Dense(32, activation="relu")(data_in)
#         data_out = layers.Dense(1, activation="linear")(dense_1)
#         self.model = keras.Model(data_in, data_out)
#         self.model.compile(optimizer=self.optimizer, loss='mape') #mse  mae optimizer='adam'
        #print('tf define done')

        INPUT_SHAPE = self.input_shape
        OUTPUT_SHAPE = self.conf['future_target']
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=INPUT_SHAPE))
        self.model.add(tf.keras.layers.LSTM(16, activation='relu'))
        self.model.add(tf.keras.layers.Dense(OUTPUT_SHAPE))#(72))
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
        return

    def fit(self, x_train, y_train, x_val, y_val):
        c = self.conf
#         self.history_callback = self.model.fit(x_train,y_train,
#                   epochs=self.conf['epochs'],
#                   batch_size=self.conf['bs'],
#                   shuffle=False,
#                   validation_data=(x_val, y_val),
#                   #callbacks=[keras.callbacks.EarlyStopping(patience=5)],
#                   verbose=0,
#                  )
        self.history_callback = self.model.fit(
            self.train_data_multi, 
            epochs=c['epochs'],
            steps_per_epoch=c['evaluation_interval'],
            validation_data=self.val_data_multi,
            validation_steps=c['validation_steps'],
        )
        print('tf fit done')
        return

    def predict(self, x_test_n):
        c = self.conf
        STEP = c['step']
        BUFFER_SIZE = c['buffer_size']
        BATCH_SIZE = c['batch_size']
        past_history = c['past_history']
        future_target = c['future_target']

        y_test_n = np.full((x_test_n.shape[0], 1), 0.5).astype('float64') # need it for fun work

        def joinback_x_y(x, y):
            dataset = np.concatenate((y,x), axis=1)
            return dataset # big np array, like original train_data df
        dataset = joinback_x_y(x_test_n, y_test_n)

        x_test_multi, y_test_multi = multivariate_data(dataset, dataset[:, 0], 0, None, past_history, future_target, STEP)
        # test_data_multi = test_data_multi.cache().batch(BATCH_SIZE).repeat()
        test_data_multi = tf.data.Dataset.from_tensor_slices((x_test_multi, y_test_multi))
        # test_data_multi = test_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        test_data_multi = test_data_multi.cache().batch(BATCH_SIZE).repeat()
        
        predictions = []
        for x, y in test_data_multi.take(x_test_n.shape[0]):
            predictions.append(self.model.predict(x)[0][0])

        # x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0, TRAIN_SPLIT, past_history, future_target, STEP)
        # x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0], TRAIN_SPLIT, None, past_history, future_target, STEP)
        # x_train_multi = multivariate_x(x, 0, past_history, future_target, STEP)
        # train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi))
        # train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        #print('tf predict done')
        return np.array(predictions).astype('float64') #self.model.predict(x)
        
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
