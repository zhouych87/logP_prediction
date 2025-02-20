from mpi4py import MPI
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Draw
import pandas as pd
import numpy as np
import math
import libset
import pickle
from pathlib import Path
from io import StringIO

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ndecimal = 0
sllist = [1.0]  # [0.03,0.05,0.1,0.2,0.5,1.0,1.5,2.0,2.5,3.0]
nsc = 500

dat = pd.read_csv('opera_without_tauts.csv')
nend = 1000
alls = dat.SMILES
y = dat.logp
dat1 = Chem.SDMolSupplier("LogP_QR.sdf")

start_idx = rank * nend
end_idx = start_idx + nend

if rank == size - 1:
    end_idx = len(alls)

my = []
for i in range(start_idx, end_idx):
    print("Processing: ",i)
    status=1
    try:
        mol=Chem.MolFromSmiles(alls[i])
    except Exception as e:
        print("Warning: Failed transfer smiles to mol",i)
        print(f"Warning: Failed transfer smiles to mol - {e}",i)
        status=-1
    if mol==None:
        status=-1
        print("Warning: Failed transfer smiles to mol",i)
        print("This data is dropped")
    if status>=0:
        xtmp0,status=libset.mol3dn(mol,sllist,nsc,ndecimal,i,status)
        #print("main pro ",i,status)  
        if status>=0:
            if xtmp0.isna().any().any(): 
                print('Try Sdf') 
                status=1
                try:
                    mol1=dat1[i]
                except Exception as e:
                    print("Warning: Failed transfer sdf to mol",i)
                    print(f"Warning: Failed transfer sdf to mol - {e}",i)
                    status=-1
                if mol1==None:
                    status=-1
                    print("Warning: Failed transfer sdf to mol",i)
                    print("This data is dropped")
                if status>=0:
                    xtmp1,status=libset.mol3dn(mol1,sllist,nsc,ndecimal,i,status)
                    if status>=0:
                        nx1=np.array(xtmp1)
                        #nx=nx.reshape((-1))
                        tnx1=nx1.T
                        ltnx1=tnx1.reshape((-1))
                        mytmp=np.hstack([i,ltnx1,y[i]])
                        my.append(mytmp)
                    else:
                        print(i," processing error")
                        print("This data is dropped")        
            else:
                nx=np.array(xtmp0)
                #nx=nx.reshape((-1))
                tnx=nx.T
                ltnx=tnx.reshape((-1))
                mytmp=np.hstack([i,ltnx,y[i]])
                my.append(mytmp)
        else:
            print(i," processing error")
            print("This data is dropped")

comm.Barrier()
print('start gathering')
all_results = comm.gather(my, root=0)

if rank == 0:
    n = np.vstack(all_results)
    ind = n[:,0]
    ind = ind.reshape(-1, 1)
    y0 = n[:,-1]
    y0 = y0.reshape(-1, 1)
    x = n[:,1:-1]
    for j in range(len(sllist)):
        len0 = nsc*9
        x0 = x[:,j*len0:j*len0+len0]
        x1 = np.hstack([ind,x0,y0])
        with open('n%ss%s.pkl'%(nsc,int(sllist[j]*100)), 'wb') as f:
            pickle.dump(x1, f)
