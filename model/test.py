import pandas as pd
import numpy as np
import os,sys,inspect,pickle,json,time,datetime,re; root = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
import random as rn

# conf = json.load(open('config_zoo.json'))
# for i in conf['lgb']['rs']:
#   print(i)
import os
logs = []
for file in os.listdir('output'):
    if file[-7:-4]=='log': 
        if file[-11:-8]=='lgb': logs.append(file)
    
df = pd.read_csv(os.path.join('output',logs[0]), index_col=0) 
for i,log in enumerate(logs):    
    if i!=0: df = df.append(pd.read_csv(os.path.join('output',log), index_col=0) )  
print(df.sort_values(by=['mape_val'])[['mape_val','rounds']])  
