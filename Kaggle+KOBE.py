
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing  import LabelEncoder

from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb



def get_date(m,n):
    h=m.split('-')
    if n=='year':
        return int(h[0])  
    elif n=='month':
        return int(h[1])    
    else :
        return int(h[2])
    
def get_index(df):
    a=[]
    for i in df.index:
        a.append(i)
    return a    


def data_feature(train,test):
    
    #game_date,season,shot_type,minutes_remaining,seconds_remaining,matchup
    for dt in [train,test]:
        dt['year']=dt['game_date'].map(lambda x: get_date(x,'year'))
        dt['month']=dt['game_date'].map(lambda x: get_date(x,'month'))
        dt['date']=dt['game_date'].map(lambda x: get_date(x,'date'))
        dt['saiji']=dt['season'].map(lambda x: int(x.split('-')[0]))
        dt['type']=dt['shot_type'].map(lambda x: 1 if '2' in x else 0)
        dt['times_remaining']=dt["minutes_remaining"]*60 + dt["seconds_remaining"]
        dt['mat']=dt['matchup'].map(lambda x : 1 if '@' in x else 0)
        
    #action_type,combined_shot_type,shot_zone_area,shot_zone_basic,shot_zone_range,opponent
    features=['action_type','opponent']
    data1=pd.concat([train,test])
    le=LabelEncoder()
    
    for col in features:
        data1[col]=le.fit_transform(data1[col])
        
    data1=pd.get_dummies(data1,columns=["combined_shot_type","shot_zone_area","shot_zone_basic","shot_zone_range"])
    train=data1.loc[get_index(train),:]
    test=data1.loc[get_index(test),:]
    
    #drop unnecessary values
    train=train.drop(["game_id","game_date","season","shot_type","minutes_remaining","seconds_remaining","lat","lon",
                    "team_id","team_name","matchup"],axis=1)
    test=test.drop(["game_id","game_date","season","shot_type","minutes_remaining","seconds_remaining","lat","lon",
                    "team_id","team_name","matchup"],axis=1)
    #Standardization
    features += ['game_event_id','loc_x','loc_y','period','times_remaining','shot_distance','month','year','date','saiji']
    data1=pd.concat([train,test])
    scaler=StandardScaler()
    
    for col in features:
        data1[col]=scaler.fit_transform(data1[col])
    
    train=data1.loc[get_index(train),:]
    test=data1.loc[get_index(test),:]
    
    return train,test


def XG_boost(train,test):
    xdata=train.drop(["shot_id","shot_made_flag"],axis=1)
    ydata=train["shot_made_flag"]
    xtrain,xtest,ytrain,ytest = train_test_split(xdata,ydata,test_size = 0.3)
    
    xgb1 = xgb.XGBClassifier(seed=1, learning_rate=0.01, n_estimators=500,silent=False, max_depth=7, subsample=0.6, colsample_bytree=0.6)
    xgb1.fit(xtrain,ytrain)
    sc=xgb1.score(xtest,ytest)
    print("xgb1 score: %s"%sc)
    
    ID=test["shot_id"]
    xtest1=test.drop(["shot_id","shot_made_flag"],axis=1)
    
    xgb1.fit(xdata,ydata)
    ytest1=xgb1.predict_proba(xtest1)
    
    submission=pd.DataFrame({"shot_id":ID,"shot_made_flag":ytest1[:,1]})
    return submission
    
    
if  __name__ == '__main__':
    
    data=pd.read_csv("E:\working directory\data.csv")
    train=data.loc[data["shot_made_flag"].notnull()]
    test=data.loc[data["shot_made_flag"].isnull()]
    
    train,test=data_feature(train,test)
    submission=XG_boost(train,test)
    submission.to_csv("E:\working directory\kobe.csv",index=False)


# In[ ]:



