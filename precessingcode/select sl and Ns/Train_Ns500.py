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

def collect_result(rdn,mae,rmse,r2,res):
    one = pd.DataFrame([[rdn,mae, rmse, r2]],columns=['RandomState','MAE', 'RMSE', 'R2'])
    res = pd.concat([res,one],ignore_index=True)
    return res

try:       
    d1 = load('/home/share/xiaojian/logp/n500s%s.pkl'%sys.argv[2])
    d2 = load('/home/share/xiaojian/logp/n500s%sp2.pkl'%sys.argv[2])
    n = np.vstack((d1,d2))
except:
    n = load('/home/share/xiaojian/logp/opt1/nsc500/sl_and_Ns500/n500s%s.pkl'%sys.argv[2])
x = n[:,1:-1]
y = n[:,-1]
x = x.reshape(len(n),-1)
nn = np.column_stack((x,y))

model_name = sys.argv[1]
model_class = getattr(sklearn.linear_model, model_name)
if isinstance(model_class(), sklearn.base.BaseEstimator):
    reg = model_class()

step = 1
start_idx = rank * step
end_idx = start_idx + step

if rank == size - 1:
    end_idx = 10

my = []
for rdn in range(start_idx,end_idx):
    n_train,n_test = train_test_split(nn,test_size=0.1,random_state=rdn)
    x_train = n_train[:,:-1]
    #x_train,scaler = nomal(x_train)
    y_train = n_train[:,-1]
    x_test = n_test[:,:-1]
    #x_test = scaler.transform(x_test)
    y_test = n_test[:,-1]
    v = reg.fit(x_train,y_train)
    y_test_predicted = reg.predict(x_test)
    r2_test = r2_score(y_test, y_test_predicted)
    mae_test = mean_absolute_error(y_test, y_test_predicted)
    rmse_test = mean_squared_error(y_test, y_test_predicted,squared=False)
    res = np.array([mae_test,rmse_test,r2_test])
    my.append(res)

comm.Barrier()
print('start gathering')
predict = comm.gather(my, root=0)

if rank == 0:
    pre = np.vstack(predict)
    all = pd.DataFrame({'mae': pre[:,0],'rmse': pre[:,1],'r2': pre[:,2]})
    all.to_csv('%s_s%s.csv'%(model_name,sys.argv[2]))