def generate_log(model, val, TARGET, speed_sec, BATCH_SIZE, split_i, SPLIT_SEED, VAL_STRATEGY, KFOLD_SPLITS, HOLDOUT_TEST_SIZE):
    import pandas as pd
    import numpy as np
    import datetime
    def ser_to_arr(y): return np.array(y).reshape(len(y),1).T
    def arr_to_onehot(y):
        if y.shape[0]==1 and y.shape[1]==1: return y
        else:
            classes_count = len(set(np.squeeze(y).tolist()))
            return np.eye(classes_count)[y.reshape(-1)].T 
    return pd.Series({
        'datetime':datetime.datetime.now(),
        'train_score':model.score(),
        'val_score':model.score(val.loc[:,val.columns!=TARGET].values.T, arr_to_onehot(ser_to_arr(val[TARGET]))),
        'speed_sec':speed_sec,
        'TARGET':TARGET,
        'BATCH_SIZE':BATCH_SIZE,
        'split_step':split_i,
        'SPLIT_SEED':SPLIT_SEED, 
        'VAL_STRATEGY':VAL_STRATEGY, 
        'KFOLD_SPLITS':KFOLD_SPLITS, 
        'HOLDOUT_TEST_SIZE':HOLDOUT_TEST_SIZE,
        'optimizer':model.optimizer,
        'DO_keep_prob':model.DO_keep_prob,
        'L2_lambd':model.L2_lambd,
        'layers_dims':model.layers_dims, 
        'num_epochs':model.num_epochs,
        'learning_rate':model.learning_rate, 
        'X':model.X.shape,
        'Y':model.Y.shape, 
        'mini_batches':len(model.mini_batches), 
        'mini_batches_first':model.mini_batches[0][0].shape[1], 
        'mini_batches_last':model.mini_batches[-1][0].shape[1], 
        'print_cost':model.print_cost,  
        'activation':model.activation, 
        'activation_l':model.activation_l, 
        'loss':model.loss, 
        'lr_decay_type':model.lr_decay_type,
        'lr_decay_rate':model.lr_decay_rate,
        'wb_init_type':model.wb_init_type, 
        'wb_init_coef':model.wb_init_coef,
        'g_checking':model.g_checking,
    })