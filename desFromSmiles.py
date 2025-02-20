from rdkit import Chem
from rdkit.Chem import AllChem ,rdMolDescriptors,Draw
import pandas as pd
import numpy as np
import math
import libset
import pickle
from pathlib import Path
from io import StringIO
import sys

ndecimal=0
sllist=[0.5]
nsc=500

fname = 'SAMPL6_molecules.csv' #'SAMPL9_molecules.csv'
dat=pd.read_csv(fname)
length=len(dat) #488good,552
y=dat.log_P
alls=dat.SMILES

my=[]
for i in range(length):
    print("Processing: ",i)
    status=1
    try:
        mol=Chem.MolFromSmiles(alls[i])
    except Exception as e:
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

name='SAMPL6_s50.pkl' #'SAMPL9_s50.pkl'
with open(name,'wb') as f:
    pickle.dump(my, f)