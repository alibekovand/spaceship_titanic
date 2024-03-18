import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

from time import strftime, localtime



df_test = pd.read_csv('./test.csv')
df = pd.read_csv('./train.csv')

df_res = pd.DataFrame()
df_res['PassengerId'] = df_test['PassengerId']



# FE
spend_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['Total_spending'] = df[spend_feats].sum(axis=1)
df['No_spending'] = (df['Total_spending']==0).astype(int)

df_test['Total_spending'] = df_test[spend_feats].sum(axis=1)
df_test['No_spending'] = (df_test['Total_spending']==0).astype(int)

def cabin_parse(df):
    df['Cabin'].fillna('np.nan/-1/np.nan',inplace=True)  # можем сплитить nan
    
    df['Cabin_deck'] = df['Cabin'].apply(lambda x: x.split("/")[0]).astype(str)
    df['Cabin_number']  = df['Cabin'].apply(lambda x: x.split("/")[1]).astype(float)
    df['Cabin_side'] = df['Cabin'].apply(lambda x: x.split("/")[2]).astype(str)

    df['Cabin_deck'] = df['Cabin_deck'].replace('np.nan', np.nan) # возвращаем nan обратно
    df['Cabin_number'] = df['Cabin_number'].replace(-1, np.nan)
    df['Cabin_side'] = df['Cabin_side'].replace('np.nan', np.nan)
    return df

df = cabin_parse(df)
df_test = cabin_parse(df_test)

def Id_parse(df):
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)   
    df['Group_size']=df['Group'].map(lambda x: df['Group'].value_counts()[x])
    df['Solo']=(df['Group_size']==1).astype(int)
    return df

df = Id_parse(df)
df_test = Id_parse(df_test)

def Name_parse(df):
    df['Name'].fillna('Unknown Unknown', inplace=True)
    
    # New feature - Surname
    df['Surname'] = df['Name'].str.split().str[-1]
    
    # New feature - Family size
    df['Family_size'] = df['Surname'].map(lambda x: df['Surname'].value_counts()[x])
    
    # Put Nan's back in (we will fill these later)
    df.loc[df['Surname']=='Unknown','Surname']=np.nan
    df.loc[df['Family_size']>100,'Family_size']=np.nan
    return df

df = Name_parse(df)
df_test = Name_parse(df_test)

# missing values

df_pre = pd.concat([df, df_test], axis=0)
drop_features = ['PassengerId', 'Name', 'Surname', 'Cabin']
df_pre = df_pre.drop(columns=drop_features)

def fill_cat_nan_setup(df, target_name, test=False):
    cat_features = ['CryoSleep', 'No_spending', 'VIP',  'Destination', 'HomePlanet', 'Family_size', 'Cabin_side', 'Cabin_deck']
    cat_features.remove(target_name)
    df_copy = df
    if not test:
        df_copy = df_copy.drop(columns=['Transported'])
    
    mask = df_copy[target_name].isna()
    df_test_nan = df_copy[mask]
    df_copy = df_copy.dropna(subset=[target_name])
    labelencoder = LabelEncoder()
    for f in cat_features:
        #df_copy[f] = labelencoder.fit_transform(df_copy[f])
        df_copy[f] = df_copy[f].astype('category')
        df_test_nan[f] = df_test_nan[f].astype('category')
    df_copy[target_name] = labelencoder.fit_transform(df_copy[target_name])
    df_copy[target_name] = df_copy[target_name].astype('int')
    return df_copy, df_test_nan, mask, labelencoder

def train_model_cat(train, test, target_name):
    cat_features = ['CryoSleep', 'No_spending', 'VIP',  'Destination', 'HomePlanet', 'Family_size', 'Cabin_side', 'Cabin_deck', 'Group_size', 'Solo']
    cat_features.remove(target_name)
    X_train = train.drop([target_name], axis=1)
    y_train = train[target_name]
    X_test = test.drop([target_name], axis=1)
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    num_class = len(y_train.unique())
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'boosting_type': 'gbdt',  
        'metric': 'multi_logloss',
        'verbose': 0
    }    
    num_round = 10
    model = lgb.train(params, train_data, num_round)
    y_pred = model.predict(X_test)#, num_iteration=model.best_iteration)
    y_pred = np.argmax(y_pred, axis=1)
    return y_pred

def fill_nan_cat(df, test=False):
    target_cat_features = ['VIP', 'HomePlanet', 'CryoSleep', 'Family_size', 'Destination', 'Cabin_deck', 'Cabin_side']
    for f in target_cat_features:
        if (test) & (f == 'Family_size'):
            continue
        df_copy, df_test_nan, mask, labelencoder = fill_cat_nan_setup(df, f, test)
        y_pred = train_model_cat(df_copy, df_test_nan, f)
        y_pred = labelencoder.inverse_transform(y_pred)
        df.loc[mask, f] = y_pred
        print(f, 'filled')

fill_nan_cat(df_pre)
df_pre.isna().sum()


def fill_cont_nan_setup(df, target_name, test=False):
    cat_features = ['CryoSleep', 'No_spending', 'VIP',  'Destination', 'HomePlanet', 'Family_size', 'Cabin_side', 'Cabin_deck', 'Group_size', 'Solo']
    #cont_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_number']
    #cont_features.remove(target_name)
    df_copy = df
    if not test:
        df_copy = df_copy.drop(columns=['Transported'])
    for f in cat_features:
        df_copy[f] = df_copy[f].astype('category')
    mask = df_copy[target_name].isna()
    df_test_nan = df_copy[mask]
    df_copy = df_copy.dropna(subset=[target_name])
    return df_copy, df_test_nan, mask

def train_model_cont(train, test, target_name):
    cat_features = ['CryoSleep', 'No_spending', 'VIP',  'Destination', 'HomePlanet', 'Family_size', 'Cabin_side', 'Cabin_deck']
    X_train = train.drop([target_name], axis=1)
    y_train = train[target_name]
    X_test = test.drop([target_name], axis=1)
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
    params = {
        'task': 'train', 
        'boosting': 'gbdt',
        'objective': 'regression',
        'num_leaves': 10,
        'learnnig_rage': 0.3,
        'metric': {'l2','l1'},
        'verbose': -1
    }
    num_round = 100
    model = lgb.train(params, train_data, num_round)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    return y_pred

def fill_nan_cont(df, test=False):
    target_cont_features = ['Cabin_number', 'FoodCourt', 'VRDeck', 'Spa', 'RoomService',  'ShoppingMall', 'Age',]
    for f in target_cont_features:
        df_copy, df_test_nan, mask = fill_cont_nan_setup(df, f, test)
        y_pred = train_model_cont(df_copy, df_test_nan, f)
        df.loc[mask, f] = y_pred

fill_nan_cont(df_pre)

print('Cont feats filled')

# recalculate spending

spend_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

df_pre['Total_spending'] = df_pre[spend_feats].sum(axis=1)
df_pre['No_spending'] = (df_pre['Total_spending']==0).astype(int)

train_idx = df_pre['Transported'].isnull()
df = df_pre.loc[~train_idx]
df_test = df_pre.loc[train_idx]
df_test = df_test.drop(columns=['Transported'])

# scaling

for f in ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Total_spending']:
    df[f]=np.log(1+df[f])
    df_test[f]=np.log(1+df_test[f])

print('Preproccessing done')
# modeling
    

target = 'Transported'
X_train = df.drop(columns=[target])
y_train = df[target]
X_test = df_test


def cat_prepare_data(X):
    cat_features = ['CryoSleep', 'Destination', 'HomePlanet', 'Family_size', 'Cabin_side', 'Cabin_deck', 'Group_size', 'No_spending', 'VIP', 'Solo']
    X[cat_features] = X[cat_features].astype(str)
    return X

def cat_best_train(X_train, y_train):
    cat_features = ['CryoSleep', 'Destination', 'HomePlanet', 'Family_size', 'Cabin_side', 'Cabin_deck', 'Group_size', 'No_spending', 'VIP', 'Solo']
    X_train = cat_prepare_data(X_train)
    params = {'loss_function':'Logloss', 
          'eval_metric':'AUC', 
          'random_seed': 42,
          #'early_stopping_rounds': 200,
          'iterations': 1800,
          'learning_rate': 0.018791,
          #'boosting_type': 'Ordered',
          #'bootstrap_type': 'MVS',
          # 'depth': 12, 'colsample_bylevel': 0.964262259559453, 'min_data_in_leaf': 92
         }

    model_cat = CatBoostClassifier(**params, silent=True)
    model_cat.fit(X_train, y_train, 
                  cat_features=cat_features,
              #eval_set=(X_valid, y_valid),
              #use_best_model=True, 
              #plot=True
             )
    model_cat.set_probability_threshold(0.5045)
    return model_cat



# submission

model_cat = cat_best_train(X_train, y_train)
#model_cat.save_model('./model/cat_model ' + strftime("%Y-%m-%d %H:%M:%S", localtime()))
model_cat.save_model('./model/catboost.cbm')
X_test_cat = cat_prepare_data(X_test)
y_pred_cat = model_cat.predict(X_test_cat)
df_res['Transported'] = y_pred_cat
df_res.to_csv('./data/results.csv', index=False)
