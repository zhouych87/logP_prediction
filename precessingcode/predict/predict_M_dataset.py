import pickle
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split

from pathlib import Path
import sys 
from sklearn.feature_selection import SelectPercentile,SelectFromModel 
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge,Lasso,BayesianRidge,ARDRegression,GammaRegressor,OrthogonalMatchingPursuitCV,RidgeCV,RANSACRegressor,TheilSenRegressor,ElasticNetCV,LassoCV,HuberRegressor,PoissonRegressor,TweedieRegressor
from sklearn.linear_model import __all__
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#data loading
def load(path):
    n = pickle.load(open(path,'rb'))
    return n

def nomal(x):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(x)
    rx = scaler.transform(x) 
    
    return rx,scaler
    
d1 = load('/home/share/xiaojian/logp/n500s50.pkl')
d2 = load('/home/share/xiaojian/logp/n500s50p2.pkl')
n_pre = load('/home/share/xiaojian/logp/opt1/nsc500/fit_s50/des/SAMPL6_s50.pkl')
npre = load('/home/share/xiaojian/logp/opt1/nsc500/fit_s50/des/SAMPL9_s50.pkl')
n = np.vstack((d1,d2))
x = n[:,1:-1]
y = n[:,-1]
x = x.reshape(len(n),-1)
nn = np.column_stack((x,y))
x_pre = n_pre[:,1:-1]
y_pre = n_pre[:,-1]
xpre = npre[:,1:-1]
ypre = npre[:,-1]
model_name = sys.argv[1]
model_class = getattr(sklearn.linear_model, model_name)
if isinstance(model_class(), sklearn.base.BaseEstimator):
    reg = model_class()

des = rank
threshold = float(sys.argv[2])
path1 = '/home/share/xiaojian/logp/opt1/nsc500/ARD_fea_s50/fea%s.pkl'%rank
imp = load(path1)

my=[]
SAM6=[]
SAM9=[]

for rdn in range(20):
    n_train,n_test = train_test_split(nn,test_size=0.25,random_state=rdn)
    x_train = n_train[:,:-1]
    #x_train,scaler = nomal(x_train)
    y_train = n_train[:,-1]
    x_test = n_test[:,:-1]
    #x_test = scaler.transform(x_test)
    y_test = n_test[:,-1]
    selected_features = imp > threshold 
    x_train0 = x_train[:,selected_features]
    print(x_train0.shape)
    x_test0 = x_test[:,selected_features]
    v = reg.fit(x_train0,y_train)
    y_test_predicted = reg.predict(x_test0)
    r2_test = r2_score(y_test, y_test_predicted)
    mae_test = mean_absolute_error(y_test, y_test_predicted)
    rmse_test = mean_squared_error(y_test, y_test_predicted,squared=False)
    res0 = np.array([mae_test,rmse_test,r2_test])
    my.append(res0)
    #x_pre = scaler.transform(x_pre)
    x_pre1 = x_pre[:,selected_features]
    y_predicted = reg.predict(x_pre1)
    r2 = r2_score(y_pre, y_predicted)
    mae= mean_absolute_error(y_pre, y_predicted)
    rmse = mean_squared_error(y_pre, y_predicted,squared=False)
    res1 = np.array([mae,rmse,r2])
    SAM6.append(res1)
    #xpre = scaler.transform(xpre)
    xpre1 = xpre[:,selected_features]
    ypredicted = reg.predict(xpre1)
    r2_1 = r2_score(ypre, ypredicted)
    mae_1= mean_absolute_error(ypre, ypredicted)
    rmse_1 = mean_squared_error(ypre, ypredicted,squared=False)
    res2 = np.array([mae_1,rmse_1,r2_1])
    SAM9.append(res2)

my = np.vstack(my)
SAM6 = np.vstack(SAM6)
SAM9 = np.vstack(SAM9)

comm.Barrier()
print('start gathering')
pre0 = comm.gather(my, root=0)
pre1 = comm.gather(SAM6, root=0)
pre2 = comm.gather(SAM9, root=0)

if rank == 0:
    test0 = np.hstack(pre0)
    SAM6_pre = np.hstack(pre1)
    SAM9_pre = np.hstack(pre2)
    avg = np.array([np.mean(test0[:,i]) for i in range(test0.shape[1])])
    columns = [f'{metric}{i}' for i in range(20) for metric in ['mae_', 'rmse_', 'r2_']]
    avg = pd.DataFrame([avg],columns=columns,index=['Average']) 
    all = pd.DataFrame(test0,columns=columns)
    all = pd.concat([all,avg])
    all.to_csv('%s_%s_alldes_0.75.csv'%(model_name,str(sys.argv[2])))
    avg1 = np.array([np.mean(SAM6_pre[:,i]) for i in range(SAM6_pre.shape[1])])
    avg1 = pd.DataFrame([avg1],columns=columns,index=['Average']) 
    all1 = pd.DataFrame(SAM6_pre,columns=columns)
    all1 = pd.concat([all1,avg1])
    all1.to_csv('%s_%s_SAMPL6_alldes_0.75.csv'%(model_name,str(sys.argv[2])))
    avg2 = np.array([np.mean(SAM9_pre[:,i]) for i in range(SAM9_pre.shape[1])])
    avg2 = pd.DataFrame([avg2],columns=columns,index=['Average']) 
    all2 = pd.DataFrame(SAM9_pre,columns=columns)
    all2 = pd.concat([all2,avg2])
    all2.to_csv('%s_%s_SAMPL9_alldes_0.75.csv'%(model_name,str(sys.argv[2])))