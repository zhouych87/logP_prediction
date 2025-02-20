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
sllist=[0.03,0.05,0.075,0.1,0.2,0.5,1.0,1.5,2.0,2.5,3.0]
nsc=100
#sllist=[1.0,1.5]
#nsc=10

alls=Chem.SDMolSupplier("LogP_QR.sdf")
dat=pd.read_csv('opera.csv')
y=dat.exp
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

mx=pd.DataFrame()
for i in range(rank,len(alls),size):
#for i in range(rank,10,size):
    mysmilse=alls[i]
    xtmp=libset.mol3dn(mysmilse,sllist,nsc,ndecimal,i) 
    index=np.array([i for j in range(len(xtmp))])
    pdindex=pd.DataFrame(index)
    pdindex.columns=['index']
    imx=pd.concat((pdindex,xtmp),axis=1)
    imx.set_index(['index'], inplace=True)
    mx=pd.concat((mx,imx))

print("mx done",rank,mx)
all_x = comm.gather(mx, root=0)
#tx=[mx1,mx]

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

