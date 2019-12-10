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


#for x, y in model.train_data_multi.take(1):
#print('x=',x.shape) # (16, 7, 2) (batch_size, past_history, features_count_with_target)
#print('y=',y.shape) # (16, 13) (batch_size, future_target)

class TrainTFLSTM():
    optimizer = None
    model = None
    history_callback = None
    conf = None
    train_data_multi = None
    val_data_multi = None
    input_shape = None
    x_train_multi = None
    y_train_multi = None
    x_val_multi = None
    y_val_multi = None

    def __init__(self, conf):
        self.conf = conf
        #c['past_history'] = 720//c['step']
        self.conf['future_target'] = self.conf['future_target']//self.conf['step']
        tf.random.set_seed(self.conf['rs'])
        initial_learning_rate = conf['initial_learning_rate']#0.001

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=conf['decay_steps'],#50,#50 100000,
            decay_rate=conf['decay_rate'],#0.96,
            staircase=True)

        if conf['optimizer'] == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        elif conf['optimizer'] == 'RMSprop':
            self.optimizer=tf.keras.optimizers.RMSprop(clipvalue=conf['clipvalue'])

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
        x_train_multi, y_train_multi = multivariate_data(dataset[:, 1:], dataset[:, 0], 0, TRAIN_SPLIT, past_history, future_target, STEP)
        x_val_multi, y_val_multi = multivariate_data(dataset[:, 1:], dataset[:, 0], TRAIN_SPLIT, None, past_history, future_target, STEP)
        print('x_train_multi',x_train_multi.shape, 'y_train_multi',y_train_multi.shape)
        print('x_val_multi',x_val_multi.shape, 'y_val_multi',y_val_multi.shape)
        # join cols (tf input format)
        train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
        train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        # train_data_multi = train_data_multi.cache().batch(BATCH_SIZE)#.repeat()
        val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
        val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
        self.train_data_multi = train_data_multi
        self.val_data_multi = val_data_multi
        self.input_shape = x_train_multi.shape[-2:]
        self.x_train_multi = x_train_multi
        self.y_train_multi = y_train_multi
        self.x_val_multi = x_val_multi
        self.y_val_multi = y_val_multi
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
        
        self.model.add(tf.keras.layers.LSTM(100, return_sequences=True, input_shape=INPUT_SHAPE))
        self.model.add(tf.keras.layers.LSTM(16, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.2))

        # self.model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=INPUT_SHAPE))
        # self.model.add(tf.keras.layers.LSTM(16, activation='relu'))

        self.model.add(tf.keras.layers.Dense(OUTPUT_SHAPE))#(72))
        # self.model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss=self.conf['metric']) #mae
        self.model.compile(optimizer=self.optimizer, loss=self.conf['metric']) #mae
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
            verbose=0,
        )
        print('tf fit done')
        return

    def predict(self, x_test_n, df=None):
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

        x_test_multi, y_test_multi = multivariate_data(dataset[:, 1:], dataset[:, 0], 0, None, past_history, future_target, STEP)
        # test_data_multi = test_data_multi.cache().batch(BATCH_SIZE).repeat()
        # test_data_multi = test_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        #test_data_multi = tf.data.Dataset.from_tensor_slices((x_test_multi, y_test_multi))
        #test_data_multi = test_data_multi.cache().batch(BATCH_SIZE).repeat()
        
        # bad long way
        # predictions = []
        # for x, y in test_data_multi.take(x_test_n.shape[0]):
        #     predictions.append(self.model.predict(x)[0][0])

        # x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0, TRAIN_SPLIT, past_history, future_target, STEP)
        # x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0], TRAIN_SPLIT, None, past_history, future_target, STEP)
        # x_train_multi = multivariate_x(x, 0, past_history, future_target, STEP)
        # train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi))
        # train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        #print('tf predict done')
        #predictions = self.model.predict(test_data_multi)
        #return np.array(predictions).astype('float64') #self.model.predict(x)

        y_test_pred = self.model.predict(x_test_multi)
        y_test_pred = y_test_pred[:,-1:] # use only last col for predictions
        filler = y_test_pred[0,0] #
        diff = x_test_n.shape[0] - y_test_pred.shape[0]
        return np.pad(y_test_pred, [(diff,0),(0,0)], 'constant', constant_values=(filler))
        
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


    def plot_1_prediction(self, date, dataset, TARGET_ORIG, USE_TEST, NORMALIZE, train, felist, filler=0): # date="2018-04-01 00:06:00"
        import matplotlib.pyplot as plt
        import datetime
        import utils
        from pandas.plotting import register_matplotlib_converters
        register_matplotlib_converters()
        model = self.model
        conf_i = self.conf

        def denorm(normed):
            if NORMALIZE: y_pred1, y_pred2, y_pred3 = utils.denormalize_by_train(normed, normed, normed, train, TARGET_ORIG, USE_TEST)
            else: y_pred1 = normed
            return y_pred1
        def norm(unnormed):
            if NORMALIZE: x_train_n, x_val_n, x_test_n, y_train_n, y_val_n = utils.normalize_by_train(unnormed, unnormed, unnormed, unnormed, unnormed, train, TARGET_ORIG, USE_TEST)
            else: x_train_n = unnormed
            return x_train_n
        def convert_date_to_id(data, d="2018-04-01 00:03:00"): return data.index.tolist().index(datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S'))
        idx = convert_date_to_id(dataset, date)
        h_s = int(conf_i['past_history'])
        f_t = int(conf_i['future_target'])
        features = felist[0]
        print('idx=',idx,'h_s=',h_s,'f_t=',f_t,'features=',features)
        
        # x y for predict
        x_train_multilike_one = norm(dataset[features].iloc[idx-h_s:idx,:].values).reshape(1, h_s, len(features))
        y_train_pred_one = denorm(model.predict(x_train_multilike_one))
        y_train_pred_one = np.pad(y_train_pred_one,[(0,0),(h_s,0)], 'constant', constant_values=(filler)).reshape(-1)
        
        # x y for true
        df = dataset.iloc[idx-h_s:idx+f_t,:]
        x_train_dates = df.index
        y_train_orig = df[TARGET_ORIG]
        
        # plot
        fig, axarr = plt.subplots()
        axarr.plot(x_train_dates, y_train_orig, label='true')
        axarr.plot(x_train_dates, y_train_pred_one, label='predicted')
        axarr.legend(loc="upper right")
        plt.show()
        return
    # model.plot_1_prediction("2018-04-23 08:07:00", train, TARGET_ORIG, USE_TEST, NORMALIZE, train, felist, filler=30) # date="2018-04-01 00:06:00"
    # model.plot_1_prediction("2018-04-23 08:25:00", val, TARGET_ORIG, USE_TEST, NORMALIZE, train, felist, filler=30) # date="2018-04-01 00:06:00"

    def _test(self, train, TARGET_ORIG, NORMALIZE):
        def denorm(normed):
            import utils
            y_pred1, y_pred2, y_pred3 = utils.denormalize_by_train(normed, normed, normed, train, TARGET_ORIG, USE_TEST)
            return y_pred1
        conf_i = self.conf  
        model = self.model          
        # y train vs tf_prepared
        h_s = int(conf_i['past_history'])
        for i in range(10):
            normed = model.y_train_multi[i,:]
            res = denorm(normed) if NORMALIZE else normed
            print('tf',res[0],'= orig:', train[TARGET_ORIG].iloc[i+h_s])
            # tf 39.453331534911435 = orig: 39.453331534911435
            # tf 41.49387020200261 = orig: 41.49387020200261
            # tf 40.17080958783588 = orig: 40.17080958783588
            
        # x train vs tf_prepared (norm doesnt work here)
        f_t = int(conf_i['future_target'])
        feat_sample = model.x_train_multi[0:1,:,:]
        print(feat_sample[:,:,-1])
        print(train[TARGET_ORIG].head(f_t))
            # [[34.03916506 35.22933916 31.79634203 36.1406496  30.99724382 32.88653873
            #   33.3395665  37.18612343 40.93511977 44.64689782]]
            # date
            # 2018-04-01 00:01:00    34.039165
            # 2018-04-01 00:02:00    35.229339
            # 2018-04-01 00:03:00    31.796342
            # 2018-04-01 00:04:00    36.140650
            # 2018-04-01 00:05:00    30.997244
        for x, y in model.train_data_multi.take(1):    
            print(y[0])            
            # tf.Tensor([37.20623761], shape=(1,), dtype=float64)
        # plot train by tf creators
        def create_time_steps(length):
            time_steps = []
            for i in range(-length, 0, 1):
                time_steps.append(i)
            return time_steps
        def multi_step_plot(history, true_future, prediction, STEP):
            plt.figure(figsize=(12, 6))
            num_in = create_time_steps(len(history))
            num_out = len(true_future)
            plt.plot(num_in, np.array(history[:, 0]), label='History') # (history[:, 1]), label='History')
            plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
                     label='True Future')
            if prediction.any():
                plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                       label='Predicted Future')
            plt.legend(loc='upper left')
            plt.show()
        for x, y in model.train_data_multi.take(1): multi_step_plot(x[0], y[0], np.array([0]), 1)            
        # pred by tf creators
        for x, y in model.val_data_multi.take(1): # take one batch
            multi_step_plot(x[0], y[0], model.model.predict(x)[0], model.conf['step'])
            print(model.model.predict(x)[0].shape)
        return
