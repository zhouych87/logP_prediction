import pickle
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error

SAMPL6_dat = pickle.load(open('SAMPL6_s50.pkl','rb'))
SAMPL6_x = SAMPL6_dat[:,1:-1]
SAMPL6_y = SAMPL6_dat[:,-1]
SAMPL9_dat = pickle.load(open('SAMPL9_s50.pkl','rb'))
SAMPL9_x = SAMPL9_dat[:,1:-1]
SAMPL9_y = SAMPL9_dat[:,-1]

threshold = 5e-2
fea_path = 'fea.pkl'
model_path = 'model.pkl'
fea_imp = pickle.load(open('fea.pkl','rb'))
reg = pickle.load(open('model.pkl','rb'))

selected_features = fea_imp > threshold

SAMPL6_x = SAMPL6_x[:,selected_features]
SAMPL6_y_predicted = reg.predict(SAMPL6_x)
r2_SAMPL6 = r2_score(SAMPL6_y, SAMPL6_y_predicted)
mae_SAMPL6 = mean_absolute_error(SAMPL6_y, SAMPL6_y_predicted)
rmse_SAMPL6 = root_mean_squared_error(SAMPL6_y, SAMPL6_y_predicted)

SAMPL9_x = SAMPL9_x[:,selected_features]
SAMPL9_y_predicted = reg.predict(SAMPL9_x)
r2_SAMPL9 = r2_score(SAMPL9_y, SAMPL9_y_predicted)
mae_SAMPL9 = mean_absolute_error(SAMPL9_y, SAMPL9_y_predicted)
rmse_SAMPL9 = root_mean_squared_error(SAMPL9_y, SAMPL9_y_predicted)

print('SAMPL6_predict: MAE=%.4f , RMSE=%.4f , R2=%.4f'%(mae_SAMPL6,rmse_SAMPL6,r2_SAMPL6))
print('SAMPL9_predict: MAE=%.4f , RMSE=%.4f , R2=%.4f'%(mae_SAMPL9,rmse_SAMPL9,r2_SAMPL9))




