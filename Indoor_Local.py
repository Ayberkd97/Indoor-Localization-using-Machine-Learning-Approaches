import pandas as pd
from pickle import TRUE
import sys, os, time
import numpy as np
import numpy.matlib
import json
from joblib import dump, load
from pathlib import Path
import io

from sklearn import svm
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import SGDRegressor 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor 
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D, Input
from keras.utils import np_utils
from pandas import read_csv
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from statistics import mean
from sklearn.ensemble import IsolationForest
from scipy.ndimage import gaussian_filter1d
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import Ridge
from sklearn import linear_model
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score 
import keras_tuner
import tensorflow as tf
from tensorflow import keras
import time

data=pd.read_csv("labeled_data.csv")
frequencies = [8e7,30e7,99e7]

def l2_dist(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)
    dx = x1 - x2
    dy = y1 - y2
    dx = dx ** 2
    dy = dy ** 2
    dists = dx + dy
    dists=dists.astype("float")
    dists = np.sqrt(dists)
    return np.mean(dists)

def moving_average(x, w):
    y=np.pad(x, (w//2, w-1-w//2), mode='edge')
    y_smooth = np.convolve(y, np.ones((w,))/w, mode='valid') 
    return y_smooth


groups_in_file = data.groupby(['position_x','position_y'])

dicts2={}
for f in frequencies:
    dicts={}
    for k,group in enumerate(groups_in_file.groups):        
        temp_filter = (data['position_x'] == group[0])&(data['position_y'] == group[1])
        temp_train = data.loc[temp_filter].loc[data['frequency'] == f].iloc[:,4:]
        dicts[group]=np.mean(temp_train["receiver_1"]),np.mean(temp_train["receiver_2"]),np.mean(temp_train["receiver_3"]),np.mean(temp_train["receiver_4"])
    dicts2[f]=dicts
    
groups_in_file = data.groupby(['position_x','position_y'])
realdicc={}
for L,groups in enumerate(groups_in_file.groups):
    realdic={}
    for f in frequencies:
        temp_filter = (data['position_x'] == groups[0])&(data['position_y'] == groups[1])
        temp_test = data.loc[temp_filter].loc[data['frequency'] == f].iloc[:,4:]
        disct={}
        for k,group in enumerate(groups_in_file.groups):
            a=dicts2[f][group]-np.mean(temp_test)
            disct[group]=np.abs(a[0])+np.abs(a[1])+np.abs(a[2])+np.abs(a[3])
        realdic[f]=sorted(disct.items(), key=lambda item: item[1])[1]
    realdicc[groups]=realdic



def preprocessing1(Outlier, Smoothing):
    
    data=pd.read_csv("labeled_data.csv")
    
    data["r12"]=data["receiver_1"]-data["receiver_2"]
    data["r13"]=data["receiver_1"]-data["receiver_3"]
    data["r14"]=data["receiver_1"]-data["receiver_4"]
    data.drop(columns=['receiver_1','receiver_2','receiver_3','receiver_4'],inplace=True)
    
    if Outlier == 1: 
        df = pd.DataFrame(columns=['position_x', 'position_y','r12', 'r13', 'r14','frequency'])
        for f in frequencies:
            for k,group in enumerate(groups_in_file.groups):
                temp_filter = (data['position_x'] == group[0])&(data['position_y'] == group[1])
                temp_train = data.loc[temp_filter].loc[data['frequency'] == f].iloc[:,4:].values
                model=IsolationForest(contamination=float(0.05))
                yhat=model.fit_predict(temp_train)
                mask = yhat != -1
                temp_train2=temp_train[mask,:]
                locations = data.loc[data['frequency'] == f].loc[temp_filter].iloc[:,2:4].values
                yy=locations[mask,:]
                dataset=pd.DataFrame({'position_x':yy[:,0], 'position_y':yy[:,1],'r12': temp_train2[:, 0], 'r13': temp_train2[:, 1], 'r14': temp_train2[:, 2], 'frequency':f})
                df=pd.concat([df, dataset])
        data=df.copy()        
    else:
        pass
    
    if Smoothing == 1:
        data["r1_2"]=moving_average(data["r12"],5)
        data["r1_3"]=moving_average(data["r13"],5)
        data["r1_4"]=moving_average(data["r14"],5)
        
        data.drop(columns=['r12','r13','r14'],inplace=True)
        
        new_cols = ["position_x","position_y","frequency","r1_2","r1_3","r1_4"]
        data=data.reindex(columns=new_cols)
        
    else:
        data.rename(columns={'r12': 'r1_2', 'r13': 'r1_3', 'r14': 'r1_4'}, inplace=True)
        new_cols = ["position_x","position_y","frequency","r1_2","r1_3","r1_4"]
        data=data.reindex(columns=new_cols)
    
    return data

import tensorflow as tf
def create_deep(inp_dim):
    seed = 7
    np.random.seed(seed)
    model = Sequential()
    model.add(Dense(60, input_dim=inp_dim, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(60, activation='relu'))
    model.add(Dropout(0.5))    
    model.add(Dense(2, activation='relu'))
    # Compile model
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['mse'])
    return model
es = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto', restore_best_weights=True)

def iterated_position_estimation1(data, Scaling, Validation):
    '''pop one position and train one AI Model
    test it with the popped position data and plot it in a heatmap'''
    
    #reg = make_pipeline(StandardScaler(),MultiOutputRegressor(SGDRegressor(average = 1,epsilon=0.1,loss="huber",max_iter = 100000,tol=0.0001,penalty='elasticnet',verbose=1)))
    #reg = make_pipeline(StandardScaler(),MultiOutputRegressor(SGDRegressor(epsilon=0.4,loss="huber",max_iter = 100000,tol=0.0001,penalty='elasticnet',verbose=1)))#linear_model.LassoLars(alpha=.1, normalize=False)
    #regr1 = make_pipeline(StandardScaler(),svm.LinearSVR(C=30, epsilon=0.01,max_iter = 100000,verbose = 1))
    #regr2 = make_pipeline(StandardScaler(),svm.LinearSVR(C=30, epsilon=0.01,max_iter = 100000,verbose = 1))
    #reg = make_pipeline(StandardScaler(),MultiOutputRegressor(svm.SVR(kernel="rbf", C=30, epsilon=0.5, max_iter = 100000,verbose = 1)))
    #reg = make_pipeline(StandardScaler(),MultiOutputRegressor(svm.LinearSVR(C=30, epsilon=0.5,max_iter = 100000,verbose = 1)))
    groups_in_file = data.groupby(['position_x','position_y'])
    l2=[]
    a=[]
    byproc=[]
    for k,group in enumerate(groups_in_file.groups):
        a.append(group)
    for k,group in enumerate(groups_in_file.groups):

        #posätäon x and position y are grouped.
        temp_filter = (data['position_x'] == a[k][0])&(data['position_y'] == a[k][1])
        
        if k==38:
            k=-1
            
        frequencies = data.loc[temp_filter].iloc[:,2].values
        frequencies = pd.Series(frequencies).drop_duplicates().tolist()
        for f in frequencies:
            reg=create_deep(4)
            if Validation == 1:
                temp_filter_not = (data['position_x'] != a[k][0])&(data['position_y'] != a[k][1])&(data['position_x'] != a[k+1][0])&(data['position_y'] != a[k+1][1])              
                val_temp_filter= (data['position_x'] == a[k+1][0])&(data['position_y'] == a[k+1][1])
            
            elif Validation == 2:
                temp_filter_not = (data['position_x'] != a[k][0])&(data['position_y'] != a[k][1])&(data['position_x'] != realdicc[(a[k][0], a[k][1])][f][0][0])&(data['position_y'] != realdicc[(a[k][0], a[k][1])][f][0][1])
                val_temp_filter= (data['position_x'] == realdicc[(a[k][0], a[k][1])][f][0][0])&(data['position_y'] == realdicc[(a[k][0], a[k][1])][f][0][1])
                
            temp_train = data.loc[(temp_filter_not)&(data['frequency'] == f)].iloc[:,2:]
            temp_val= data.loc[(val_temp_filter)&(data['frequency'] == f)].iloc[:,2:]            
            temp_test = data.loc[(temp_filter)&(data['frequency'] == f)].iloc[:,2:]
            
            locations = data.loc[(data['frequency'] == f)&(temp_filter)].iloc[:,0:2].values
            locations_val = data.loc[(data['frequency'] == f)&(val_temp_filter)].iloc[:,0:2].values
            locations_not = data.loc[(data['frequency'] == f)&(temp_filter_not)].iloc[:,0:2].values

            if Scaling == 1:
                PredictorScaler=StandardScaler()
                PredictorScalerFit=PredictorScaler.fit(temp_train)
                temp_train=PredictorScalerFit.transform(temp_train)
                temp_val=PredictorScalerFit.transform(temp_val)
                temp_test=PredictorScalerFit.transform(temp_test)
                
            elif Scaling == 2:
                PredictorScaler=MinMaxScaler()
                PredictorScalerFit=PredictorScaler.fit(temp_train)
                temp_train=PredictorScalerFit.transform(temp_train)
                temp_val=PredictorScalerFit.transform(temp_val)
                temp_test=PredictorScalerFit.transform(temp_test)
            else:
                pass
    
            temp_train, locations_not = shuffle(temp_train, locations_not, random_state=0)
        
            temp_train = np.asarray(temp_train).astype(np.float32)
            locations_not = np.asarray(locations_not).astype(np.float32)
            temp_val = np.asarray(temp_val).astype(np.float32)
            locations_val = np.asarray(locations_val).astype(np.float32)
            temp_test = np.asarray(temp_test).astype(np.float32)
            locations = np.asarray(locations).astype(np.float32)   
            
            hist=reg.fit(temp_train,locations_not,validation_data=(temp_val, locations_val), epochs=30, batch_size=100,  callbacks = [es])

            temp_pred = reg.predict(temp_test)
            print("Prediced Locations")
            print(temp_pred)
            print("Real Locations")
            print(locations)
            print(k,group,f, l2_dist((temp_pred[:,0],temp_pred[:,1]), (locations[:,0],locations[:,1])))
            l2.append(l2_dist((temp_pred[:,0],temp_pred[:,1]), (locations[:,0],locations[:,1])))

    loss=np.mean(l2)
    print(loss)            

def iterated_position_estimation2(data,reg, Scaling):
    '''pop one position and train one AI Model
    test it with the popped position data and plot it in a heatmap'''
        
    #reg = make_pipeline(StandardScaler(),MultiOutputRegressor(SGDRegressor(average = 1,epsilon=0.1,loss="huber",max_iter = 100000,tol=0.0001,penalty='elasticnet',verbose=1)))
    #reg = make_pipeline(StandardScaler(),MultiOutputRegressor(SGDRegressor(epsilon=0.4,loss="huber",max_iter = 100000,tol=0.0001,penalty='elasticnet',verbose=1)))#linear_model.LassoLars(alpha=.1, normalize=False)
    #regr1 = make_pipeline(StandardScaler(),svm.LinearSVR(C=30, epsilon=0.01,max_iter = 100000,verbose = 1))
    #regr2 = make_pipeline(StandardScaler(),svm.LinearSVR(C=30, epsilon=0.01,max_iter = 100000,verbose = 1))
    #reg = make_pipeline(StandardScaler(),MultiOutputRegressor(svm.SVR(kernel="rbf", C=30, epsilon=0.5, max_iter = 100000,verbose = 1)))
    #reg = make_pipeline(StandardScaler(),MultiOutputRegressor(svm.LinearSVR(C=30, epsilon=0.5,max_iter = 100000,verbose = 1)))
    groups_in_file = data.groupby(['position_x','position_y'])
    l2=[]
    a=[]
    for k,group in enumerate(groups_in_file.groups):
        a.append(group)
    for k,group in enumerate(groups_in_file.groups):

        #posätäon x and position y are grouped.
        temp_filter = (data['position_x'] == group[0])&(data['position_y'] == group[1])
            
        frequencies = data.loc[temp_filter].iloc[:,2].values
        frequencies = pd.Series(frequencies).drop_duplicates().tolist()
        for f in frequencies:
            reg=reg
            
            temp_filter_not = (data['position_x'] != group[0])&(data['position_y'] != group[1])

            temp_train = data.loc[(temp_filter_not)&(data['frequency'] == f)].iloc[:,3:]
            temp_test = data.loc[(temp_filter)&(data['frequency'] == f)].iloc[:,3:]
            
            locations = data.loc[(data['frequency'] == f)&(temp_filter)].iloc[:,0:2].values
            locations_not = data.loc[(data['frequency'] == f)&(temp_filter_not)].iloc[:,0:2].values

            #regr1.fit(temp_train,x_coordinates_not)
            #regr2.fit(temp_train,y_coordinates_not)
            
            if Scaling == 1:
                PredictorScaler=StandardScaler()
                PredictorScalerFit=PredictorScaler.fit(temp_train)
                temp_train=PredictorScalerFit.transform(temp_train)
                temp_test=PredictorScalerFit.transform(temp_test)
                
            elif Scaling == 2:
                PredictorScaler=MinMaxScaler()
                PredictorScalerFit=PredictorScaler.fit(temp_train)
                temp_train=PredictorScalerFit.transform(temp_train)
                temp_test=PredictorScalerFit.transform(temp_test)
            else:
                pass

            temp_train, locations_not = shuffle(temp_train, locations_not, random_state=0)
            
            temp_train = np.asarray(temp_train).astype(np.float32)
            locations_not = np.asarray(locations_not).astype(np.float32)
            temp_test = np.asarray(temp_test).astype(np.float32)
            locations = np.asarray(locations).astype(np.float32)
    
            hist=reg.fit(temp_train,locations_not)
            #temp_pred_x = regr1.predict(temp_test)
            #temp_pred_y = regr2.predict(temp_test)
            temp_pred = reg.predict(temp_test)
            print("Prediced Locations")
            print(temp_pred)
            print("Real Locations")
            print(locations)
            print(k,group,f, l2_dist((temp_pred[:,0],temp_pred[:,1]), (locations[:,0],locations[:,1])))
            l2.append(l2_dist((temp_pred[:,0],temp_pred[:,1]), (locations[:,0],locations[:,1])))
    
    loss=np.mean(l2)
    print(loss)
    print(l2)

def iterated_position_estimation4(data, Scaling):
    '''pop one position and train one AI Model
    test it with the popped position data and plot it in a heatmap'''
        
    #reg = make_pipeline(StandardScaler(),MultiOutputRegressor(SGDRegressor(average = 1,epsilon=0.1,loss="huber",max_iter = 100000,tol=0.0001,penalty='elasticnet',verbose=1)))
    #reg = make_pipeline(StandardScaler(),MultiOutputRegressor(SGDRegressor(epsilon=0.4,loss="huber",max_iter = 100000,tol=0.0001,penalty='elasticnet',verbose=1)))#linear_model.LassoLars(alpha=.1, normalize=False)
    #regr1 = make_pipeline(StandardScaler(),svm.LinearSVR(C=30, epsilon=0.01,max_iter = 100000,verbose = 1))
    #regr2 = make_pipeline(StandardScaler(),svm.LinearSVR(C=30, epsilon=0.01,max_iter = 100000,verbose = 1))
    #reg = make_pipeline(StandardScaler(),MultiOutputRegressor(svm.SVR(kernel="rbf", C=30, epsilon=0.5, max_iter = 100000,verbose = 1)))
    #reg = make_pipeline(StandardScaler(),MultiOutputRegressor(svm.LinearSVR(C=30, epsilon=0.5,max_iter = 100000,verbose = 1)))
    groups_in_file = data.groupby(['position_x','position_y'])
    l2=[]
    a=[]
    for k,group in enumerate(groups_in_file.groups):
        a.append(group)
    for k,group in enumerate(groups_in_file.groups):

        #posätäon x and position y are grouped.
        temp_filter = (data['position_x'] == group[0])&(data['position_y'] == group[1])
        
        for f in frequencies:
            reg=create_deep(4)
            
            temp_filter_not = (data['position_x'] != group[0])&(data['position_y'] != group[1])

            temp_train = data.loc[(temp_filter_not)&(data['frequency'] == f)].iloc[:,2:]
            temp_test = data.loc[(temp_filter)&(data['frequency'] == f)].iloc[:,2:]
            
            locations = data.loc[(data['frequency'] == f)&(temp_filter)].iloc[:,0:2].values
            locations_not = data.loc[(data['frequency'] == f)&(temp_filter_not)].iloc[:,0:2].values

            #regr1.fit(temp_train,x_coordinates_not)
            #regr2.fit(temp_train,y_coordinates_not)
            
            if Scaling == 1:
                PredictorScaler=StandardScaler()
                PredictorScalerFit=PredictorScaler.fit(temp_train)
                temp_train=PredictorScalerFit.transform(temp_train)
                temp_test=PredictorScalerFit.transform(temp_test)
                
            elif Scaling == 2:
                PredictorScaler=MinMaxScaler()
                PredictorScalerFit=PredictorScaler.fit(temp_train)
                temp_train=PredictorScalerFit.transform(temp_train)
                temp_test=PredictorScalerFit.transform(temp_test)
            else:
                pass

            temp_train, locations_not = shuffle(temp_train, locations_not, random_state=0)
            
            temp_train = np.asarray(temp_train).astype(np.float32)
            locations_not = np.asarray(locations_not).astype(np.float32)
            temp_test = np.asarray(temp_test).astype(np.float32)
            locations = np.asarray(locations).astype(np.float32)
            
            hist=reg.fit(temp_train,locations_not, epochs=30, batch_size=100)
            #temp_pred_x = regr1.predict(temp_test)
            #temp_pred_y = regr2.predict(temp_test)
            temp_pred = reg.predict(temp_test)
            print("Prediced Locations")
            print(temp_pred)
            print("Real Locations")
            print(locations)
            print(k,group,f, l2_dist((temp_pred[:,0],temp_pred[:,1]), (locations[:,0],locations[:,1])))
            l2.append(l2_dist((temp_pred[:,0],temp_pred[:,1]), (locations[:,0],locations[:,1])))
    
    loss=np.mean(l2)
    print(loss)
    print(l2)
iterated_position_estimation2(preprocessing1(1,1),DecisionTreeRegressor(criterion='friedman_mse', splitter='random', max_depth=20, min_samples_leaf=15),1)
iterated_position_estimation2(preprocessing1(1,1),RandomForestRegressor(n_estimators=100, max_features=1, max_depth=15, min_samples_leaf=15),1)
iterated_position_estimation2(preprocessing1(1,1),MultiOutputRegressor(svm.SVR(kernel='rbf', C=5.0, epsilon=0.01, gamma =0.1)),1)
iterated_position_estimation2(preprocessing1(1,1),xgb.XGBRegressor(max_depth=5, learning_rate=0.1, colsample_bytree=0.5, subsample=0.5),1)
   
iterated_position_estimation1(preprocessing1(1,1),1,2)
iterated_position_estimation4(preprocessing1(1,1),1)
