def create_one_hot_encodings(df, col_name='Embarked', drop_original=False): 
    '''
    creates n new colls (binary), n = classes count
    boolean values create only 1 new col - 1/0
    can work not only with strings, but with ints too (Pclass=1/2/3 -> 3 new binary cols)
    '''
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    list_array_series = df[col_name]; #print(list_array_series)
    int_encoded = LabelEncoder().fit_transform(list_array_series); #print(int_encoded)
    onehot_encoded = OneHotEncoder(sparse=False, categories='auto').fit_transform(int_encoded.reshape(len(int_encoded), 1))
    classes_count = onehot_encoded.shape[1]
    if classes_count > 2:
        for class_num in range(classes_count): df[col_name+'_'+str(class_num)] = onehot_encoded[:,class_num].astype(int)
    else: df[col_name+'_'+str(0)] = onehot_encoded[:,1].astype(int)
    if drop_original: df.drop([col_name], axis=1, level=None, inplace=True, errors='raise')
    return


def generate_fruits_dataframes():
    import pandas as pd
    train = pd.DataFrame({'fruit_label':[0,    1,    0,    1,    0,    1,    0,    1,    0,    1,    0,    1], 
                          'mass':       [154,  260,  148,  218,  150,  142,  148,  316,  148,  148,  152,  198],
                          'width':      [6.15, 7.55, 6.55, 7.05, 5.95, 5.97, 5.85, 8.52, 6.45, 6.08, 6.35, 6.87],
                          'color_score':[35,   46,   34,   46,   27,   52,   28,   49,   28,   60,   28,   64]})
    test  = pd.DataFrame({'fruit_label':[0,    1,    0,    1,    0,    1,    0,    1,    0,    1,    0,    1], 
                          'mass':       [150,  120,  154,  148,  154,  196,  154,  164,  150,  142,  144,  262],
                          'width':      [6.15, 5.53, 6.65, 6.17, 7.15, 6.48, 6.35, 5.93, 5.75, 6.08, 5.70, 7.58],
                          'color_score':[24,   47,   31,   46,   42,   58,   27,   49,   19,   61,   26,   50]})
    val   = pd.DataFrame({'fruit_label':[0,    1,    0,    1,    0,    1],
                          'mass':       [150,  172,  150,  154,  146,  288],
                          'width':      [5.55, 6.42, 6.45, 6.37, 6.25, 8.15],
                          'color_score':[27,   55,   18,   60,   29,   52]})
    return train, test, val




def read_fruits_dataframes_from_txt(CLASSES_COUNT=999999):
    import pandas as pd
    import numpy as np
    df = pd.read_table('datasets/fruit_data_my.txt')
    df = df[['fruit_label', 'mass', 'width', 'color_score']]
    df['fruit_label'] = df['fruit_label'].replace(4, 0)
    # shuffle rows:
    np.random.seed(1)
    permutation = list(np.random.permutation(len(df)))  # [24, 7, 37 ... 148 elements from 1 to 148] # np.random.permutation(4) => # array([0, 3, 2, 1])
    df = df.iloc[permutation]                # (12288, 148) # select all pixels, select all cols but using new order from permutation list 
    df.reset_index(inplace=True, drop=True) # indes now from 0 again
    # split df
    train = df.copy().iloc[:20] # 0 - 19 len=20
    test = df.copy().iloc[20:40] # 20 - 39  len=20
    val = df.copy().iloc[40:] # 40 - 59  len=20
    # reduce classes count to CLASSES_COUNT
    train = train[train['fruit_label']<CLASSES_COUNT] # if CLASSES_COUNT=2 we use only classes 0 and 1
    test = test[test['fruit_label']<CLASSES_COUNT]
    val = val[val['fruit_label']<CLASSES_COUNT]
    return train, test, val

def df_to_inputs(df, target, include_vec=True): 
    '''train -> x_train, y_train, y_train_vec  / val ->  x_val, y_val, y_val_vec
    Example: x_train, y_train, y_train_vec = df_to_inputs(train, TARGET)'''
    import numpy as np
    def series_to_array(y): return np.array(y).reshape(len(y),1).T
    def convert_to_one_hot(y): return np.eye(len(set(np.squeeze(y).tolist())))[y.reshape(-1)].T # classes count = len(set(np.squeeze(Y).tolist()))
    x_df = df.drop(target, axis=1, level=None, inplace=False, errors='raise').values.T
    y_df_vec = series_to_array(df[target])
    y_df_onehot = convert_to_one_hot(y_df_vec)
    if include_vec: return x_df, y_df_onehot, y_df_vec
    else:           return x_df, y_df_onehot
    # np.eye(4)[np.array([0,1])]
    # array([[1., 0., 0., 0.],
    #    [0., 1., 0., 0.]])

def ser_to_arr(y): return np.array(y).reshape(len(y),1).T
def arr_to_onehot(y):
    if y.shape[0]==1 and y.shape[1]==1: return y
    else:
        classes_count = len(set(np.squeeze(y).tolist()))
        return np.eye(classes_count)[y.reshape(-1)].T 
