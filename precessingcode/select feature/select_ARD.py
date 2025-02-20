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
       
d1 = load('/home/share/xiaojian/logp/n500s50.pkl')
d2 = load('/home/share/xiaojian/logp/n500s50p2.pkl')
n = np.vstack((d1,d2))
x = n[:,1:-1]
y = n[:,-1]
x = x.reshape(len(n),-1)
nn = np.column_stack((x,y))

step = 1
start_idx = rank * step
end_idx = start_idx + step

if rank == size - 1:
    end_idx = 20

for rdn in range(start_idx,end_idx):
    n_train,n_test = train_test_split(nn,test_size=0.1,random_state=rdn)
    x_train = n_train[:,:-1]
    y_train = n_train[:,-1]
    x_test = n_test[:,:-1]
    y_test = n_test[:,-1]
    alg = ARDRegression()
    selector = SelectFromModel(estimator=alg,threshold=1e-20).fit(x_train, y_train)
    feature_importances=np.array(abs(selector.estimator_.coef_))
    with open('fea%s.pkl'%rdn,'wb') as f:
        pickle.dump(feature_importances,f)
