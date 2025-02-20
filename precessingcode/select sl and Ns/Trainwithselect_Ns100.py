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

sllist = [3,5,7,10,20,50,100,150,200,250,300]
file0 = sllist[rank]     
n = load('/home/share/xiaojian/logp/s%s.pkl'%file0)
x = n[:,1:-1]
y = n[:,-1]
x = x.reshape(len(n),-1)
nn = np.column_stack((x,y))
model_name = sys.argv[1]
model_class = getattr(sklearn.linear_model, model_name)
if isinstance(model_class(), sklearn.base.BaseEstimator):
    reg = model_class()

threshold = float(sys.argv[2])
path1 = '/home/share/xiaojian/logp/opt1/nsc100/ARD_fea/fea0.pkl'
imp = load(path1)

my0=[]
my=[]

for rdn in range(10):
    n_train,n_test = train_test_split(nn,test_size=0.1,random_state=rdn)
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
    my0.append(res0)

test0 = np.vstack(my0)
avg = np.array([np.mean(test0[:,i]) for i in range(test0.shape[1])])
my.append(avg)

comm.Barrier()
print('start gathering')
pre0 = comm.gather(my, root=0)

if rank == 0:
    test1 = np.vstack(pre0)
    columns = ['mae_mean', 'rmse_mean', 'r2_mean']
    all = pd.DataFrame(test1,columns=columns)
    sll = pd.DataFrame(sllist,columns=['sl'])
    all = pd.concat([sll,all],axis=1)
    all.to_csv('%s_select.csv'%model_name)