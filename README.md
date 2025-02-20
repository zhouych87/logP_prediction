# A new simple and efficient molecular descriptor for the fast and accurate prediction of log P   

Zeng et al. J. Mater. Inf. 2025, 5, 4  
DOI: 10.20517/jmi.2024.61  

## How to use this code to predict log P of SAMPL6 and SAMPL9, or your own molecules.

### Prepare a csv file  
A csv file includes at least two columns with column names of "SMILES" and "log_P". 
If you do not know the log_p, just input any number. Then, the predicted MAE, RMSE nad R2 are meaningless.  
Assuming the input file is SAMPL6_molecules.csv  

### run code

```
python logPpredict.py SAMPL6_molecules.csv
```

>(molp) zhouych@node19:~/software/logp$ python logPpredict.py SAMPL6_molecules.csv  
Processing:  0  
Processing:  1  
Processing:  2  
Processing:  3  
Processing:  4  
Processing:  5  
Processing:  6  
Processing:  7  
Processing:  8  
Processing:  9  
Processing:  10  
Predicted log P:  
[4.428 3.892 2.937 2.927 3.05  1.514 3.58  3.27  2.144 2.615 2.852]  
Predict: MAE=0.2689 , RMSE=0.3094 , R2=0.7845  
>


## The folder of processingcode is how we build and train the model.

