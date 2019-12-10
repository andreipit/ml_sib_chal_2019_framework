
# split cols, to numpy
def split_cols_np(train, val, test, TARGET):
    x_train = train.drop(TARGET, axis=1, inplace=False).values # faster if target first col: train.iloc[:, 1:].values
    x_val   = val.drop(TARGET, axis=1, inplace=False).values   # val.iloc[:, 1:].values
    x_test  = test.values
    y_train = train[[TARGET]].values
    y_val   = val[[TARGET]].values
    return x_train, x_val, x_test, y_train, y_val

#x_train, x_val, x_test, y_train, y_val = utils.split_cols(train, val, test, TARGET)
#print('x_train',x_train.shape, 'y_train',y_train.shape, 'x_val',x_val.shape, 'y_val',y_val.shape, 'x_test',x_test.shape)


def normalize_by_train(x_train, x_val, x_test, y_train, y_val, train, TARGET):
    # normalize
    #center, scale = train.iloc[:, 1:].mean().values, train.iloc[:, 1:].std().values
    center_x, scale_x = train.drop(TARGET, axis=1, inplace=False).mean().values, train.drop(TARGET, axis=1, inplace=False).std().values
    center_y, scale_y = train[TARGET].mean(), train[TARGET].std()
    x_train_n = (x_train - center_x)/scale_x
    x_val_n   = (x_val   - center_x)/scale_x
    x_test_n  = (x_test  - center_x)/scale_x
    y_train_n = (y_train - center_y)/scale_y
    y_val_n   = (y_val   - center_y)/scale_y
    #print('x_train_n',x_train_n.shape, 'y_train_n',y_train_n.shape, 'x_val_n',x_val_n.shape, 'y_val_n', y_val_n.shape)
    return x_train_n, x_val_n, x_test_n, y_train_n, y_val_n


def denormalize_by_train(y_train_pred_n, y_val_pred_n, y_test_pred_n, train, TARGET):
    y_train_pred = y_train_pred_n * train[TARGET].std() + train[TARGET].mean()
    y_val_pred   = y_val_pred_n   * train[TARGET].std() + train[TARGET].mean() # cause center/scale made from train only
    y_test_pred  = y_test_pred_n  * train[TARGET].std() + train[TARGET].mean() # cause center/scale made from train only
    return y_train_pred, y_val_pred, y_test_pred

# def normalize(x_train, x_val, x_test, y_train, y_val, train, TARGET):
#   # normalize
#   #center, scale = train.iloc[:, 1:].mean().values, train.iloc[:, 1:].std().values

#   center_x, scale_x = train.drop(TARGET, axis=1, inplace=False).mean().values, train.drop(TARGET, axis=1, inplace=False).std().values
#   center_y, scale_y = train[TARGET].mean(), train[TARGET].std()

#   x_train_n = (x_train - center)/scale
#   x_val_n   = (x_val   - center)/scale
#   x_test_n  = (x_test  - center)/scale
#   y_train_n = (y_train - train[TARGET].mean())/train[TARGET].std()
#   y_val_n   = (y_val   - train[TARGET].mean())/train[TARGET].std()
#   #print('x_train_n',x_train_n.shape, 'y_train_n',y_train_n.shape, 'x_val_n',x_val_n.shape, 'y_val_n', y_val_n.shape)
#   return x_train_n, x_val_n, x_test_n, y_train_n, y_val_n 


def split_model_config(conf_mtype): # in: model params with lists, out: list of params-combination
    if conf_mtype['tune']=='': return [conf_mtype]
    import itertools; model_configs = []; list_of_lists = []
    def tunelists_to_floats(conf_mtype, comb):
        conf_i = conf_mtype.copy()
        for i,t in enumerate(conf_mtype['tune'].split(',')): conf_i[t] = comb[i]
        return conf_i
    # 1 build list of lists
    for t in conf_mtype['tune'].split(','): list_of_lists.append(conf_mtype[t])
    # 2 get all combinations of a list of lists
    combs = list(itertools.product(*list_of_lists)) # [(2734, 10), (2734, 20), (2734, 30), (1, 10), (1, 20), (1, 30), (2, 10), (2, 20), (2, 30)]
    # 3 fill separate model_config from each combination 
    for comb in combs:model_configs.append(tunelists_to_floats(conf_mtype, comb))
    return model_configs
    #print(split_model_config(conf['lgb']))



def print_log():
    import pandas as pd
    import json
    conf = json.load(open('config_zoo.json'))
    for mtype in conf['models'].split(','):
        #mtype = 'lgb' #'tf','lgb'
        if mtype=='tf': start_char = -10
        elif mtype=='lgb': start_char = -11   

        import os
        import pandas as pd
        logs = []
        for file in os.listdir('output'):
            if file[-7:-4]=='log': 
                #if file[-11:-8]=='lgb': logs.append(file)
                if file[start_char:-8]==mtype: logs.append(file)
        df = pd.read_csv(os.path.join('output',logs[0]), index_col=0) 
        for i,log in enumerate(logs):    
            if i!=0: df = df.append(pd.read_csv(os.path.join('output',log), index_col=0) )  
        #print(df.sort_values(by=['mape_val'])[['mape_val','rounds']].head())
        print(df.sort_values(by=['mape_val'])[['mape_val']+conf[mtype]['tune'].split(',')].head(10))    



#     mape_train = '{:.2f}'.format(100 * mape(train[TARGET].values, y_train_pred_p.values)) #print(f"MAPE (training  set): {100 * mape(train[TARGET].values, y_train_pred_p.values):.2f}%")
#     mape_test = '{:.2f}'.format(100 * mape(val[TARGET].values,   y_val_pred_p.values)) #print(f"MAPE (cross-val set): {100 * mape(val[TARGET].values,   y_val_pred_p.values):.2f}%")
#     print(f"MAPE (training  set): {mape_train}%")
#     print(f"MAPE (cross-val set): {mape_test}%")
    #score_train = mape(train[TARGET].values, y_train_pred_p.values)
    #score_val   = mape(val[TARGET].values,   y_val_pred_p.values)
        

def plot_loss(mtype, count):
    import matplotlib.pyplot as plt
    import pandas as pd
    import json
    log = pd.read_csv('output/log_'+mtype+'.csv', index_col=0)#.iloc[:count]#.iloc[-1]['history']

    cols = 2
    rows = log.shape[0]//cols + log.shape[0]%cols

    fig, axarr = plt.subplots(rows+1, cols, figsize=(12, 5*rows))
    i = 0
    for row in range(rows):
        for col in range(cols):
            if i==log.shape[0]: break
            line = pd.read_csv('output/log_'+mtype+'.csv', index_col=0).iloc[i]
            his_id = line.name
            his_str = line['history']
            his_dict = json.loads(his_str.replace("'", "\""))
            #print(his_dict)
            #print(len(his_dict['ox_list']), len(his_dict['oy_train_list']), len(his_dict['oy_val_list']))
            axarr[row][col].set_title(his_dict['title']); 
            axarr[row][col].set_xlabel('epochs of ' + his_id)
            axarr[row][col].set_ylabel('error')
            axarr[row][col].plot(his_dict['ox_list'], his_dict['oy_train_list'], label='train')
            axarr[row][col].plot(his_dict['ox_list'], his_dict['oy_val_list'], label='val')
            axarr[row][col].legend(loc="upper right")
            i+=1

    # plt.ylabel('cost'); 
    # plt.xlabel('epochs'); 
    # plt.title(''); 
    # plt.legend(['train','val'])
    plt.subplots_adjust(hspace=0.6, wspace=0.2)
    plt.show()  

# def plot_costs(ox_list, y_train_list, y_val_list, title):
#     import matplotlib.pyplot as plt
#     %matplotlib inline
#     plt.plot(ox_list, y_train_list); 
#     plt.plot(ox_list, y_val_list); 
#     plt.ylabel('cost'); 
#     plt.xlabel('epochs'); 
#     plt.title(title); 
#     plt.legend(['train','val'])
#     plt.show()  
# #plot_costs(ox=[x for x in range(1,conf_i['epochs']+1)], train_list=history_callback.history["loss"], val_list=history_callback.history["val_loss"])
# plot_costs(*model.get_costs())

