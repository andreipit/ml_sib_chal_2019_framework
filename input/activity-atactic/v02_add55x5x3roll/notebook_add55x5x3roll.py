#!/usr/bin/env python
# coding: utf-8

# ### 0) mini config

# In[1]:


INPUT_FOLDER = '\\v00_original'
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'
FORMAT = '.pkl' # '.zip' '.pkl'


# ### 1) libs and consts

# In[2]:


import os,sys,inspect,pickle
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
gragrandparentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir))) 
sys.path.insert(0,gragrandparentdir) # pipeline-template
import time; tic = time.time() #print(time.time()-tic,'sec')
import tqdm
import numpy as np
import pandas as pd
import config
import kglpipe
import os #;sys.exit(sys.path) # sys.exit(np.array([5,52,12]))
path = gragrandparentdir + kglpipe.myconfig['PATH'] + INPUT_FOLDER # path = os.path.join(gragrandparentdir, kglpipe.myconfig['PATH']+INPUT_FOLDER)
encoding = 'utf-8'
global_target = 0


# ### 2) read

# In[3]:


train_data = pickle.load( open(os.path.join(path, TRAIN_FILENAME+FORMAT), "rb"))
test_data = pickle.load( open(os.path.join(path, TEST_FILENAME+FORMAT), "rb"))
#activity_test_target, atactic_test_target = load_stamps(DATA_DIR/'v00_original')
#activity_test_target = pd.read_csv(DATA_DIR.joinpath("activity_test_timestamps.csv"), index_col="date", parse_dates=["date"])
#atactic_test_target = pd.read_csv(DATA_DIR.joinpath("atactic_test_timestamps.csv"), index_col="date", parse_dates=["date"])
activity_test_target = pd.read_csv(os.path.join(path, "activity_test_timestamps.csv"), index_col="date", parse_dates=["date"])

print('train',train_data.shape, 'test',test_data.shape, 'activity_test_target',activity_test_target.shape)#, 'atactic_test_target',atactic_test_target.shape)


# In[4]:


print(list(train_data.index)[:1],list(train_data.index)[-1:])
print(list(test_data.index)[:1],list(test_data.index)[-1:])


# In[5]:


test_data.index[0]


# ### 3) update
#     step 1 'train' -----------------------------------------------------------------
#     (463058, 60)  train           with 4 targets           (feb2018-dec2018)
#     (463058, 4)   train_targets  (activity has 308313 not nulls)
#     (103651, 56)  test           without 4 targets         (jan2019-mar2019)
#     (566709, 55)  data           without 4 targets and f28
#                    +
#     (566709, 825) all_features    has its unique 825 (55x3x5) columns, generated from data
#                    ||
#     (566709, 880) full_data = data + all_features = without 4 targets and f28
# 
#     (463058, 881)  activity_train_with_nulls = train_targets[["activity"]] + full_data - rows
#                                463058 x1                       566709 x880  
# 
#     (300737, 881) activity_train  (without rows with any NaN) save as train.pkl
#     (103651, 880) activity_test  = full_data from jan2019 => useless! save as test_useless.pkl
#     
#     step 2 'test' -----------------------------------------------------------------
#     (85891,    0) activity_test_target (empty csv with dates: jan2019-mar2019)
#                     + (join, fillna)
#     (566709, 880) full_data (see in top)
#                     ||
#     (85891,  880) test_activity_data  => real test! save as test.pkl
#     
#     conclusion: use only activity, add 825 cols to train, delete null rows and col f28
#         * for testing - predict full_data from jan2019
#         * for trainig - split activity_data on train/dev

# In[6]:


data = pd.concat([train_data[test_data.columns], test_data]) ;print('data',data.shape) # use all columns except 4 targets
train_targets = train_data[["activity", "atactic_1", "atactic_2", "atactic_3"]].copy() ;print('train_targets',train_targets.shape)


# In[7]:


series = train_targets[['activity']].notnull().all(axis=1)
print(len(series[series==True]), 'rows without any null in whole row') # 300759 


# In[8]:


# generate 825 new cols from joineddata

data.drop("f28", axis=1, inplace=True)
#ACOLS = ["atactic_1", "atactic_2", "atactic_3"]
#not_null_atactic = train_targets.loc[train_targets[ACOLS].notnull().all(axis=1), ACOLS] ;print(not_null_atactic.shape)
PERIODS = ["1H", "3H", "6H"]
AGGREGATES = ["mean", "median", "std", "max", "min"]
#PERIODS = ["3H"]
#AGGREGATES = ["mean"]
all_features = []
for period in tqdm.tqdm_notebook(PERIODS):
    for agg in AGGREGATES:
        print(period,agg,end=',')
        rolling_features = data.rolling(period).aggregate(agg)
        rolling_features.rename(lambda x: "_".join([x, period, agg]), axis=1, inplace=True)
        all_features.append(rolling_features)
all_features = pd.concat(all_features, axis=1) 
print('all_features',all_features.shape) #825/5/3 = 55 #15 new cols for each of 55 features 


# In[9]:


full_data = data.join(all_features)
print('full_data',full_data.shape)


# In[10]:


# add col 'activity' from train to our data (train+test joined generated). 463058, cause in train so
activity_train_with_nulls = train_targets[["activity"]].join(full_data.shift(6, freq="H")) # add 6 hours empty rows to start, delete last 6 hours rows
print('activity_train_with_nulls',activity_train_with_nulls.shape)


# In[11]:


series = activity_train_with_nulls.notnull().all(axis=1)
print(len(series[series==True]), 'rows without any null in whole row') # 300759 


# In[12]:


activity_train = activity_train_with_nulls[activity_train_with_nulls.notnull().all(axis=1)]
print('activity_train',activity_train.shape)


# In[13]:


activity_test = full_data[full_data.index >= test_data.index[0]]
print('activity_test',activity_test.shape)


# In[14]:


test_activity_data = activity_test_target.join(full_data.shift(6, freq="H")).ffill() # Synonym for DataFrame.fillna() with method='ffill'.
print('test_activity_data',test_activity_data.shape) # test_activity_data (85891, 880)


# ### 4) save
# 

# In[15]:


activity_train.to_pickle(os.path.join(currentdir, "train.pkl"))
activity_test.to_pickle(os.path.join(currentdir, "test_useless.pkl"))
test_activity_data.to_pickle(os.path.join(currentdir, "test.pkl"))

print(time.time()-tic,'sec') # 173.91042184829712 sec


# ### 5) test

# In[16]:


# import time
# tic = time.time() 
# train_fun = pickle.load( open(os.path.join(currentdir, TRAIN_FILENAME+FORMAT), "rb"))
# test_fun = pickle.load( open(os.path.join(currentdir, TEST_FILENAME+FORMAT), "rb"))
# print('train_fun',train_fun.shape, 'test_fun',test_fun.shape)#, 'activity_test_timestamps',activity_test_timestamps.shape, 'atactic_test_timestamps',atactic_test_timestamps.shape)
# # train_fun (300737, 881) test_fun (103651, 880)
# print(time.time()-tic,'sec') # 27 sec
# # Split
# tr_data = train_fun[:"2018-10-13"]
# cv_data = train_fun["2018-10-14":]
# print('tr_data', tr_data.shape, 'cv_data',cv_data.shape) # tr_data (216400, 881) cv_data (84337, 881)

