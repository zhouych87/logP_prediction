# conda activate molp
from rdkit import Chem
from rdkit.Chem import AllChem ,rdMolDescriptors,Draw
import pandas as pd
import numpy as np
import math
import libset
import pickle
from pathlib import Path
from io import StringIO

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#rank = comm.rank

ndecimal=0
sllist=[0.03,0.05,0.075,0.1,0.2] #,0.5,1.0,1.5,2.0,2.5,3.0]
nsc=500
#sllist=[1.0,1.5]
#nsc=10

alls=Chem.SDMolSupplier("LogP_QR.sdf")
dat=pd.read_csv('opera.csv')
nend=len(alls)
nstart=10000
#nend=10000 #488good,552
y=dat.exp[:nend]
#m=alls[1]
#m.GetProp('LogP')
#nsmiles=int(np.ceil(len(alls)/size))
'''
mx1=pd.DataFrame()
for i in range(30,40,size):
    mysmilse=alls[i]
    xtmp=libset.mol3dn(mysmilse,sllist,nsc,ndecimal,i) 
    index=np.array([i for j in range(len(xtmp))])
    pdindex=pd.DataFrame(index)
    pdindex.columns=['index']
    imx=pd.concat((pdindex,xtmp),axis=1)
    imx.set_index(['index'], inplace=True)
    mx1=pd.concat((mx1,imx))

'''

my=[]
#rank=796
#for i in range(rank,len(alls),size):
for i in range(rank+nstart,nend,size):
    #print("Processing: ",i)
    status=1
    try:
        mol=alls[i]
    except Exception as e:# 如果发生异常，打印警告而不是终止程序
        print("Warning: Failed transfer sdf to mol",i)
        print(f"Warning: Failed transfer sdf to mol - {e}",i)
        status=-1
    if mol==None:
        status=-1
        print("Warning: Failed transfer sdf to mol",i)
        print("This data is dropped")
    if status>=0:
        xtmp,status=libset.mol3dn(mol,sllist,nsc,ndecimal,i,status)
        #print("main pro ",i,status)
        if status>=0:
            nx=np.array(xtmp)
            #nx=nx.reshape((-1))
            tnx=nx.T
            ltnx=tnx.reshape((-1))
            mytmp=np.hstack([i,ltnx,y[i]])
            if len(my)==0:
                my=mytmp.copy()
            else:
                my=np.vstack([my,mytmp])
        else:
            print(i," processing error")
            print("This data is dropped")

print("mx done",rank)
'''
for i in range(size):
    comm.Barrier()
    if i==rank:
        print("mx done",rank)
        #print(mx)
'''
comm.Barrier()
if rank == 0: 
    print("start gathering")


my_bytes = pickle.dumps(my)
# 使用comm.gather收集数据
gathered_data = comm.gather(my_bytes, root=0)

if rank == 0:
    gathernp=[]
    for data in gathered_data:
        np_recv = pickle.loads(data)
        #print(df_recv)
        if len(gathernp)==0:
            gathernp=np_recv.copy()
        else:
            gathernp=np.vstack([gathernp,np_recv])
    print("gathernp: ")
    print(gathernp)
    index=gathernp[:,0]
    x=gathernp[:,1:-1]
    y=gathernp[:,-1]
    rx=x.reshape(len(x),len(sllist),-1)
    for i in range(len(sllist)):
        sl=sllist[i]
        print('Start Descriptor sl=',sl)
        name='n500s'+str(int(sl*100))
        slx=rx[:,i,:]
        ltnx=slx.reshape((len(x),-1))
        mysl=np.column_stack([index,ltnx,y])
        #np.savetxt(name+'.csv', mysl, delimiter=',', fmt='%.6f')
        with open(name+'p2.pkl','wb') as f:
            pickle.dump(mysl, f)

#tx=[mx1,mx]
'''
if rank == 0:
    # Combine the gathered DataFrames into a single DataFrame
    x = pd.concat(all_x) # ignore_index=True
    x=x.sort_index()
    print(x)
    for sl in sllist:
        print('Start Descriptor sl=',sl)
        name='s'+str(int(sl*100))
        if ndecimal>0:
            nfile='nf'+str(ndecimal)+'sl'+str(sl)+".pkl"
        else:
            nfile='allsl'+str(sl)+".pkl"
        nx=x[name]
        nx=np.array(nx)
        nx=nx.reshape((len(alls),-1))
        ny=np.array(y)
        n=np.column_stack((nx,ny))
        np.savetxt('npsave.csv', n, delimiter=',', fmt='%.5f')
        #wn=pd.DataFrame()
        #print(pd.DataFrame(n))
        #with open(nfile,'wb') as f:
        #    pickle.dump(n, f)

'''