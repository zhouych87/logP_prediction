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

def add_name(d):
    fnames=[]
    d = d.reshape(1,-1)
    for a in ['MorU','MorM','MorV','MorP','MorE','MorC','MorIP','MorIS','MorRC']:
        for s in range(500):
            fnames.append(a+'-'+str(s))
    fnames = np.array(fnames)
    print(fnames.shape)
    print(d.shape)
    n=pd.DataFrame(d,columns=fnames)
    return n
      
d1 = load('/home/share/xiaojian/logp/n500s50.pkl')
d2 = load('/home/share/xiaojian/logp/n500s50p2.pkl')
n = np.vstack((d1,d2))
x = n[:,1:-1]
y = n[:,-1]
x = x.reshape(len(n),-1)
nn = np.column_stack((x,y))

model_name = sys.argv[1]
model_class = getattr(sklearn.linear_model, model_name)
if isinstance(model_class(), sklearn.base.BaseEstimator):
    reg = model_class()

threshold = float(sys.argv[2])

step = 1
start_idx = rank * step
end_idx = start_idx + step

if rank == size - 1:
    end_idx = 20

my = []
my0 = [] 
for rdn in range(start_idx,end_idx):
    n_train,n_test = train_test_split(nn,test_size=0.1,random_state=rdn)
    x_train = n_train[:,:-1]
    #x_train,scaler = nomal(x_train)
    y_train = n_train[:,-1]
    x_test = n_test[:,:-1]
    #x_test = scaler.transform(x_test)
    y_test = n_test[:,-1]
    path1 = '/home/share/xiaojian/logp/opt1/nsc500/ARD_fea_s50/fea%s.pkl'%rdn
    fea = load(path1)
    fea = np.array(fea)
    al = add_name(fea)
    imp = al.loc[0]
    selected_features = imp > threshold
    print(al)
    print(selected_features.shape)
    feas = al.loc[:, selected_features]
    print(feas)
    x_train0 = x_train[:,selected_features]
    list0 = np.array(x_train0.shape)
    my0.append(list0)
    x_test0 = x_test[:,selected_features]
    v = reg.fit(x_train0,y_train)
    y_test_predicted = reg.predict(x_test0)
    r2_test = r2_score(y_test, y_test_predicted)
    mae_test = mean_absolute_error(y_test, y_test_predicted)
    rmse_test = mean_squared_error(y_test, y_test_predicted,squared=False)
    res = np.array([mae_test,rmse_test,r2_test])
    my.append(res)

comm.Barrier()
print('start gathering')
predict = comm.gather(my, root=0)
shape = comm.gather(my0, root=0)

if rank == 0:
    pre = np.vstack(predict)
    shape0 = np.vstack(shape)
    all = pd.DataFrame({'mae': pre[:,0],'rmse': pre[:,1],'r2': pre[:,2],'mol_num': shape0[:,0],'fea_num': shape0[:,1]})
    all.to_csv('%s_%s_ARD.csv'%(model_name,str(sys.argv[2])))
    
    
    
