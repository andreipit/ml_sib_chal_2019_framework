#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os,sys,inspect,pickle,json,time,datetime,re; root = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
import random as rn
import utils

conf = json.load(open('config_zoo.json'))
TARGET = 'activity'
DATA_DIR = 'activity-atactic'
EXTRA_DATA1 = 'v00_original\\activity_test_timestamps.csv'
rn.seed(conf['rs'])
np.random.seed(conf['rs'])

# load
train_data = pickle.load(open(os.path.join(root,'input',DATA_DIR,conf['data'],'train.pkl'),"rb"))
test_data  = pickle.load(open(os.path.join(root,'input',DATA_DIR,conf['data'],'test.pkl'),"rb"))

# load extra data
extra_data1 = pd.read_csv(os.path.join(root,'input',DATA_DIR,EXTRA_DATA1), index_col="date", parse_dates=["date"])
#activity_test_target = pd.read_csv(os.path.join(root,'input',DATA_DIR,'v00_original','activity_test_timestamps.csv'), index_col="date", parse_dates=["date"])
print('train',train_data.shape, 'test_data',test_data.shape, 'extra_data1',extra_data1.shape)#, 'atactic_test_target',atactic_test_target.shape,time.time()-tic,'sec')

# split rows
if conf['datasize']=='min':
    train = train_data[:"2018-03-13"]#[:"2018-12-13"]#[:"2018-10-13"]
    val   = train_data["2018-03-14":"2018-03-24"]#["2018-12-14":]#["2018-10-14":]
elif conf['datasize']=='med':
    train = train_data[:"2018-10-13"]
    val   = train_data["2018-10-14":]
elif conf['datasize']=='max':
    train = train_data
    val   = train_data
#train = train_data[:conf['train_till']]#[:"2018-12-13"]#[:"2018-10-13"]
#val   = train_data[conf['val_from']:conf['val_till']]#["2018-12-14":]#["2018-10-14":]
test  = test_data

# split cols to numpy and norm
x_train, x_val, x_test, y_train, y_val = utils.split_cols_np(train, val, test, TARGET) #;print('x_train',x_train.shape, 'y_train',y_train.shape, 'x_val',x_val.shape, 'y_val',y_val.shape, 'x_test',x_test.shape)
x_train_n, x_val_n, x_test_n, y_train_n, y_val_n = utils.normalize_by_train(x_train, x_val, x_test, y_train, y_val, train, TARGET) #;print('x_train_n',x_train_n.shape, 'y_train_n',y_train_n.shape, 'x_val_n',x_val_n.shape, 'y_val_n', y_val_n.shape)

def evaluate(y_train_pred_n, y_val_pred_n, y_test_pred_n, conf_i, submit=True):
    # denormalize
    y_train_pred, y_val_pred, y_test_pred = utils.denormalize_by_train(y_train_pred_n, y_val_pred_n, y_test_pred_n, train, TARGET)

    # convert to pandas
    y_train_pred_p = pd.Series(y_train_pred.flatten(), index=train.index, name=TARGET+"_pred").sort_index()
    y_val_pred_p   = pd.Series(y_val_pred.flatten(),   index=val.index,   name=TARGET+"_pred").sort_index()
    y_test_pred_p  = pd.DataFrame(y_test_pred, columns=[TARGET], index=extra_data1.index) # activity_test_target

    # evaluate
    def mape(y_true, y_pred): return 100 * np.mean(np.abs((y_pred-y_true)/y_true))
    conf_i['score_train']= mape(train[TARGET].values, y_train_pred_p.values)
    conf_i['score_val']  = mape(val[TARGET].values,   y_val_pred_p.values)

    # submit
    curr_time = re.sub(r'^(.*)-(.*)-(.*) (.*):(.*):(.*)\.(.*)$',r'y\1_\2m_\3d_\4h_\5min_\6s_\7_',str(datetime.datetime.now()))
    y_test_pred_p.loc[test.index].to_csv(os.path.join('output',curr_time+mtype+'_sub.csv'))

    # add log to log_model.csv file
    log_file = 'log_'+mtype+'.csv'
    log_df = pd.DataFrame([conf_i], index=[curr_time])
    if log_file in os.listdir('output'): log_df = pd.read_csv(os.path.join('output', log_file), index_col=0).append(log_df)   
    log_df.to_csv(os.path.join('output', log_file))
    # add log to separate file
    #pd.DataFrame([conf_i], index=[curr_time]).to_csv(os.path.join('output',curr_time+mtype+'_log.csv'))
    return
#evaluate(y_train_pred_n, y_val_pred_n, y_test_pred_n, conf['lgb'], submit=True)


# In[2]:


for mtype in conf['models'].split(','):
    for conf_i in utils.split_model_config(conf[mtype]):
        if mtype=='tf':
            from train_tf import TrainTF
            model = TrainTF(conf_i)
        elif mtype=='lgb':
            from train_lgb import TrainLGB
            model = TrainLGB(conf_i)
        model.compile(x_train.shape)
        model.fit(conf_i, x_train_n, y_train_n, x_val_n, y_val_n)
        y_train_pred_n = model.predict(x_train_n) 
        y_val_pred_n   = model.predict(x_val_n) 
        y_test_pred_n  = model.predict(x_test_n) 
        evaluate(y_train_pred_n, y_val_pred_n, y_test_pred_n, conf_i, submit=True)


# In[3]:


mtype = 'lgb'
print(pd.read_csv('output/log_'+mtype+'.csv', index_col=0)[['score_train','score_val']+conf[mtype]['tune'].split(',')].sort_values(by=['score_val']))
#pd.read_csv('output/'+mtype+'_log.csv')


# In[4]:


mtype = 'tf'
print(pd.read_csv('output/log_'+mtype+'.csv', index_col=0)[['score_train','score_val']+conf[mtype]['tune'].split(',')].sort_values(by=['score_val']))
#pd.read_csv('output/'+mtype+'_log.csv')


# In[ ]:




