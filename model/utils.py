# feature groups
rashod_r1 = ['f2','f3','f4','f5','f20','f22']
inside_r1_triple = ['f17','f25','f29','f32','f46','f47','f41']
inside_r1_double = ['f7','f8','f9']
inside_r1_unique = ['f39']

rashod_r2 = ['f10','f11','f12','f13','f21','f23']
inside_r2_triple = ['f18','f26','f31','f34','f51','f52','f50']
inside_r2_double = ['f14','f15','f16']

inside_r3_triple = ['f19','f27','f30','f33','f48','f49','f45']
inside_r3_unique = ['f36','f40']


weights = ['f53','f54','f55']
weather = ['f42','f43','f44']
temps_pro_wat_d219 = ['f1','f24','f38']
pressures_pro_az_etil = ['f6','f35','f37']
donor_in_nefras_level = ['f0']
bad = ['f28']
stars = ['f17','f18','f39','f50','f41','f38','f40','f45','f19'] # pred from activity gives e-2
rects = ['f52','f51','f46','f47','f48','f49','f36'] # pred from activity gives e-1

# groups of groups
rashod_r1_r2 = rashod_r1 + rashod_r2
inside_r1_r2 = inside_r1_triple + inside_r1_double + inside_r1_unique + inside_r2_triple + inside_r2_double
inside_r3 = inside_r3_triple + inside_r3_unique
unknown = weights + weather + temps_pro_wat_d219 + pressures_pro_az_etil + donor_in_nefras_level
together = rashod_r1_r2 + inside_r1_r2 + inside_r3 + unknown

features_dict = {'rashod_r1':rashod_r1, 'inside_r1_triple':inside_r1_triple, 'inside_r1_double':inside_r1_double,
    'inside_r1_unique':inside_r1_unique, 'rashod_r2':rashod_r2, 'inside_r2_triple':inside_r2_triple, 'inside_r2_double':inside_r2_double,
    'inside_r3_triple':inside_r3_triple, 'inside_r3_unique':inside_r3_unique, 'weights':weights, 'weather':weather,
    'temps_pro_wat_d219':temps_pro_wat_d219, 'pressures_pro_az_etil':pressures_pro_az_etil, 'donor_in_nefras_level':donor_in_nefras_level,
    'bad':bad, 'stars':stars, 'rects':rects, 'rashod_r1_r2':rashod_r1_r2, 'inside_r1_r2':inside_r1_r2, 'inside_r3':inside_r3, 
    'unknown':unknown, 'together':together}


def plot_predictions(start, stop, train, val, y_train_pred_p, y_val_pred_p, title):
    import matplotlib.pyplot as plt
    val["activity"][start:stop].plot(c="greenyellow")
    train["activity"][start:stop].plot(c="lightskyblue")
    y_train_pred_p[start:stop].plot(c="forestgreen")
    y_val_pred_p[start:stop].plot(c="orangered")
    plt.legend(loc=0)
    plt.tight_layout()
    plt.title(title, fontsize=12)
    return

def evaluate(model, train, val, test, y_test_pseudo,  extra_data1, y_train_pred_n, y_val_pred_n, y_test_pred_n, conf_i, mtype, TARGET, NORMALIZE, USE_TEST=False, submit_trainval=False, save_model=False, smooth_pred=False):
    import pandas as pd
    import numpy as np
    import re, datetime, os, pickle
    import tensorflow as tf
    if NORMALIZE: y_train_pred, y_val_pred, y_test_pred = denormalize_by_train(y_train_pred_n, y_val_pred_n, y_test_pred_n, train, TARGET, USE_TEST)
    else: y_train_pred, y_val_pred, y_test_pred = y_train_pred_n, y_val_pred_n, y_test_pred_n

    y_train_pred_p = pd.Series(y_train_pred.flatten(), index=train.index, name=TARGET+"_pred").sort_index()
    y_val_pred_p   = pd.Series(y_val_pred.flatten(),   index=val.index,   name=TARGET+"_pred").sort_index()
    if smooth_pred:
        y_train_pred_p = y_train_pred_p.subtract(train[TARGET].mean()).multiply(5/train[TARGET].std()).add(train[TARGET].mean())
        y_train_pred_p = y_train_pred_p.ewm(span = 6).mean()
        y_val_pred_p = y_val_pred_p.subtract(train[TARGET].mean()).multiply(5/train[TARGET].std()).add(train[TARGET].mean())
        y_val_pred_p = y_val_pred_p.ewm(span = 6).mean()

    if USE_TEST: y_test_pred_p  = pd.DataFrame(y_test_pred, columns=[TARGET], index=extra_data1.index) # activity_test_target
    # 1 evaluate
    def mape(y_true, y_pred): return 100 * np.mean(np.abs((y_pred-y_true)/y_true))
    conf_i['score_train']= mape(train[TARGET].values, y_train_pred_p.values)
    conf_i['score_val']  = mape(val[TARGET].values,   y_val_pred_p.values)
    conf_i['score_test_pseudo']  = mape(y_test_pseudo[TARGET].values, y_test_pred_p[TARGET].values) if USE_TEST else 0

    # 2 submit
    curr_time = re.sub(r'^(.*)-(.*)-(.*) (.*):(.*):(.*)\.(.*)$',r'y\1_\2m_\3d_\4h_\5min_\6s_\7_',str(datetime.datetime.now()))
    if USE_TEST: y_test_pred_p.loc[test.index].to_csv(os.path.join('output',curr_time+mtype+'_sub.csv'))
    print('scores of',curr_time,' score_train =',conf_i['score_train'], 'score_val =',conf_i['score_val'], 'score_test_pseudo=',conf_i['score_test_pseudo'])
    # 3 save log
    log_file = 'log_'+mtype+'.csv'
    log_df = pd.DataFrame([conf_i], index=[curr_time])
    if log_file in os.listdir('output'): log_df = pd.read_csv(os.path.join('output', log_file), index_col=0).append(log_df, sort=True)   
    log_df.to_csv(os.path.join('output', log_file))
    if submit_trainval:
        y_train_pred_p2  = pd.DataFrame(y_train_pred, columns=[TARGET], index=train.index)
        y_val_pred_p2  = pd.DataFrame(y_val_pred, columns=[TARGET], index=val.index)
        y_train_pred_p2.to_csv(os.path.join('output','trainval',curr_time+mtype+'_train_sub.csv'))
        y_val_pred_p2.to_csv(os.path.join('output','trainval',curr_time+mtype+'_val_sub.csv'))
    if save_model: 
        #pickle.dump( model, open( os.path.join('output','pretrained_models',curr_time+mtype+'_model.pkl'), "wb" ) )
        #model.to_pickle(os.path.join('output','pretrained_models',curr_time+mtype+'_model.pkl'))
        #tf.saved_model.save(model.model, os.path.join('output','pretrained_models',curr_time+mtype+'_model'))
        path_h5 = os.path.join('output','pretrained_models',curr_time+mtype+'_model.h5')
        model.model.save(path_h5)
    return

def add_log_info(conf_i, model, conf, TARGET, FEATURES, fename):
    conf_i['history'] = model.get_costs() #model.history_callback
    conf_i['datasize'] = conf['datasize']
    conf_i['data'] = conf['data']
    conf_i['data_predict'] = conf['data_predict']
    conf_i['rs_global'] = conf['rs']
    conf_i['target'] = TARGET
    conf_i['features'] = FEATURES
    conf_i['fegroup'] = fename
    conf_i['split_type'] = conf['split_type']
    conf_i['normalize'] = conf['normalize']
    return conf_i

def init_model(mtype, conf_i):
    if mtype=='tf':
        from train_tf import TrainTF
        model = TrainTF(conf_i)
    elif mtype=='lgb':
        from train_lgb import TrainLGB
        model = TrainLGB(conf_i)
    elif mtype=='xgb':
        from train_xgb import TrainXGB
        model = TrainXGB(conf_i)        
    elif mtype=='mynn':
        from train_mynn import TrainMYNN
        model = TrainMYNN(conf_i)
    elif mtype=='tf_lstm':
        from train_tf_lstm import TrainTFLSTM
        model = TrainTFLSTM(conf_i)
    elif mtype=='manual':
        from train_manual import TrainMANUAL
        model = TrainMANUAL(conf_i)
    return model

def split_rows(train_data, conf, conf_i):

    # con = '{2018-03-18 00:01:00,2018-04-18 00:01:00};{2018-04-18 00:01:00,2018-05-07 00:01:00}'
    # con = con.replace('{','').replace('}','')
    # train_dates, val_dates = con.split(';')

    fold = conf_i['fold'].replace('{','').replace('}','')
    train_dates, val_dates = fold.split(';')

    train_start, train_stop = train_dates.split(',')[0], train_dates.split(',')[1]
    val_start,   val_stop   = val_dates.split(',')[0],   val_dates.split(',')[1]

    train = train_data[train_start:train_stop]#[:"2018-12-13"]#[:"2018-10-13"]
    val   = train_data[val_start:val_stop]#["2018-12-14":]#["2018-10-14":]

    # if conf['split_type']=='dates':
    #     if conf['datasize']=='min':
    #         train = train_data[:"2018-03-13"]#[:"2018-12-13"]#[:"2018-10-13"]
    #         val   = train_data["2018-03-14":"2018-03-24"]#["2018-12-14":]#["2018-10-14":]
    #     elif conf['datasize']=='med':
    #         train = train_data[:"2018-10-13"]
    #         val   = train_data["2018-10-14":]
    #     elif conf['datasize']=='max':
    #         train = train_data
    #         val   = train_data
    # elif conf['split_type']=='division':
    #     if conf['datasize']=='min':
    #         train = train_data[:"2018-03-13"]#[:"2018-12-13"]#[:"2018-10-13"]
    #         val   = train_data["2018-03-14":"2018-03-24"]#["2018-12-14":]#["2018-10-14":]
    #     elif conf['datasize']=='med':
    #         train = train_data.iloc[:train_data.shape[0]*3//4,:]
    #         val   = train_data.iloc[train_data.shape[0]*3//4:,:]
    #     elif conf['datasize']=='max':
    #         train = train_data
    #         val   = train_data
    return train, val

# split cols, to numpy
def split_cols_np(train, val, test, TARGET, USE_TEST):
    import numpy as np
    x_train = train.drop(TARGET, axis=1, inplace=False).values # faster if target first col: train.iloc[:, 1:].values
    x_val   = val.drop(TARGET, axis=1, inplace=False).values   # val.iloc[:, 1:].values
    x_test  = test.values if USE_TEST else np.array([])
    y_train = train[[TARGET]].values
    y_val   = val[[TARGET]].values
    return x_train, x_val, x_test, y_train, y_val

#x_train, x_val, x_test, y_train, y_val = utils.split_cols(train, val, test, TARGET)
#print('x_train',x_train.shape, 'y_train',y_train.shape, 'x_val',x_val.shape, 'y_val',y_val.shape, 'x_test',x_test.shape)


def normalize_by_train(x_train, x_val, x_test, y_train, y_val, train, TARGET, USE_TEST):
    import numpy as np
    # normalize
    #center, scale = train.iloc[:, 1:].mean().values, train.iloc[:, 1:].std().values
    center_x, scale_x = train.drop(TARGET, axis=1, inplace=False).mean().values, train.drop(TARGET, axis=1, inplace=False).std().values
    center_y, scale_y = train[TARGET].mean(), train[TARGET].std()
    x_train_n = (x_train - center_x)/scale_x
    x_val_n   = (x_val   - center_x)/scale_x
    x_test_n  = (x_test  - center_x)/scale_x if USE_TEST else np.array([])
    y_train_n = (y_train - center_y)/scale_y
    y_val_n   = (y_val   - center_y)/scale_y
    #print('x_train_n',x_train_n.shape, 'y_train_n',y_train_n.shape, 'x_val_n',x_val_n.shape, 'y_val_n', y_val_n.shape)
    return x_train_n, x_val_n, x_test_n, y_train_n, y_val_n


def denormalize_by_train(y_train_pred_n, y_val_pred_n, y_test_pred_n, train, TARGET, USE_TEST):
    import numpy as np
    y_train_pred = y_train_pred_n * train[TARGET].std() + train[TARGET].mean()
    y_val_pred   = y_val_pred_n   * train[TARGET].std() + train[TARGET].mean() # cause center/scale made from train only
    y_test_pred  = y_test_pred_n  * train[TARGET].std() + train[TARGET].mean() if USE_TEST else np.array([])# cause center/scale made from train only
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
                if file[start_char:-8]==mtype: logs.append(file, sort=True)
        df = pd.read_csv(os.path.join('output',logs[0]), index_col=0) 
        for i,log in enumerate(logs):    
            if i!=0: df = df.append(pd.read_csv(os.path.join('output',log), index_col=0), sort=True )  
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
    #log = pd.read_csv('output/log_'+mtype+'.csv', index_col=0)#.iloc[:count]#.iloc[-1]['history']
    log = pd.read_csv('output/log_'+mtype+'.csv', index_col=0).tail(count)#.iloc[:count]#.iloc[-1]['history']

    cols = 2
    rows = log.shape[0]//cols + log.shape[0]%cols

    fig, axarr = plt.subplots(rows, cols, figsize=(12, 3*rows))
    i = 0
    for row in range(rows):
        for col in range(cols):
            if i==log.shape[0]: break
            #line = pd.read_csv('output/log_'+mtype+'.csv', index_col=0).iloc[i]
            line = log.iloc[i]
            his_id = line.name
            his_str = line['history']
            his_dict = json.loads(his_str.replace("'", "\""))
            #print(his_dict)
            #print(len(his_dict['ox_list']), len(his_dict['oy_train_list']), len(his_dict['oy_val_list']))
            axarr[row][col].set_title(his_dict['title']); 
            axarr[row][col].set_xlabel('epochs of ' + his_id  + ', ' + str(round(line.score_train,2)) + '/' + str(round(line.score_val,2)))
            axarr[row][col].set_ylabel('error')
            axarr[row][col].plot(his_dict['ox_list'], his_dict['oy_train_list'], label='train')
            axarr[row][col].plot(his_dict['ox_list'], his_dict['oy_val_list'], label='val')
            axarr[row][col].legend(loc="upper right")
            i+=1

    # plt.ylabel('cost'); 
    # plt.xlabel('epochs'); 
    # plt.title(''); 
    # plt.legend(['train','val'])
    plt.subplots_adjust(hspace=0.6, wspace=0.7)
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

