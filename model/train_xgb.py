import random as rn
#import lightgbm as lgb
import xgboost as xgb # xgboost-0.90
import numpy as np
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train
#"objective": "regression", # https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters 
# config params description: https://xgboost.readthedocs.io/en/latest/parameter.html

class TrainXGB():
    model = None
    conf = None
    history_callback = None

    def __init__(self, conf):
        self.conf = conf
        #print('lgb init done')
        return
        
    def prepare_data(self, x_train, y_train, x_val, y_val): # norm np.arrays
        return

    def compile(self, x_train_shape):
        #print('lgb define done')
        return

    def fit(self, x_train, y_train, x_val, y_val):
        xgb_train = xgb.DMatrix(x_train, label=y_train.reshape(-1)) # lgb_train = lgb.Dataset(x_train, label=y_train.reshape(-1))
        xgb_val = xgb.DMatrix(x_val, label=y_val.reshape(-1)) # lgb_val = lgb.Dataset(x_val, label=y_val.reshape(-1))


        evals_result = {}
        c=self.conf
        #params = {'metric':'mape', 'num_threads': -1, 'objective': 'regression', 'verbosity': 1} # rmse
        params = {
            'metric':c['metric'], 
            'num_threads':c['num_threads'], 
            'objective': c['objective'], 
            'verbosity': c['verbosity'], 
            'is_training_metric': True,
            'lambda_l2':c['lambda_l2'],
            'lambda_l1':c['lambda_l1'],
            'min_gain_to_split':c['min_gain_to_split'],
            'num_leaves':c['num_leaves'],
            'min_child_weight':c['min_child_weight'],
            'lambda':c['lambda'],
            'alpha':c['alpha'],
        } # rmse
        rounds = c['rounds']
        # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html

        self.model = xgb.train(
            params = params, 
            dtrain = xgb_train, #train_set = lgb_train, 
            num_boost_round = rounds, 
            evals = [(xgb_train, 'train'), (xgb_val, 'eval')],#valid_sets = [lgb_train,lgb_val], 
            verbose_eval=c['verbose_eval'], 
            early_stopping_rounds=c['early_stopping_rounds'], 
            callbacks=[xgb.callback.record_evaluation(evals_result)] #[lgb.record_evaluation(evals_result)]#same -> evals_result=evals_result,
            #callbacks=[lgb.print_evaluation(period=1, show_stdv=True)] #same -> evals_result=evals_result,
        )
        steps = self.model.best_iteration #print(f"Best: {steps}") #print('lgb fit done')
        self.history_callback = evals_result
        return

    def predict(self, x, df=None):
        #print('lgb predict done')
        #return self.model.predict(x)
        return self.model.predict(xgb.DMatrix(x))

    def get_costs(self):
        oy_train_list = self.history_callback['train']['rmse']#['training']['mape']
        oy_train_list = list(np.array(oy_train_list, dtype=np.float32))
        oy_val_list = self.history_callback['eval']['rmse']#['valid_1']['mape']
        oy_val_list = list(np.array(oy_val_list, dtype=np.float32))
        #ox_list = [x for x in range(1,self.conf['rounds']+1)] # early stopping can cause error
        ox_list = [x for x in range(1,len(oy_train_list)+1)]


        title = self.conf['name'] + '_'
        for t in self.conf['tune'].split(','): 
            if t!='': title += t+'_'+str(self.conf[t])+'_'

        # if tune_param!='' and len(tune_param.split(','))==1: title_unique = tune_param+'='+str(self.conf[tune_param])
        # else: title_unique = 'min_gain_to_split='+str(self.conf['min_gain_to_split']) # 'lgb_rounds='+str(self.conf['rounds'])+
        # title = title_unique# + '_score_train='+str(round(self.conf['score_train'],2)) + '_score_val='+str(round(self.conf['score_val'],2))
        #title = str(self.conf)
        return {'ox_list':ox_list, 'oy_train_list':oy_train_list, 'oy_val_list':oy_val_list, 'title':title}







