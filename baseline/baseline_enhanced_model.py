import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

df = pd.read_csv('stratified_sample.csv')

#linear regression model
#features and target for prediction
features = ['brightness', 'bright_t31', 'scan', 'track']
target = 'frp'

X = df[features]
y = df[target]

#split 10% for testing 70% for training 20% for validation
X_temp, X_test_lr, y_temp, y_test_lr = train_test_split(X, y, test_size=0.1, random_state=42)

#take  about 22% to get 20% of the original total for validation
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.222, random_state=42)

#create the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

#predictions made for validation
val_predictions = lr_model.predict(X_val)

#predictions made for testing
lr_predictions = lr_model.predict(X_test_lr)

#calculate metrics
lr_r2 = r2_score(y_test_lr, lr_predictions)
lr_mse = mean_squared_error(y_test_lr, lr_predictions)
lr_rmse = np.sqrt(lr_mse)


#xgboost model
#set frp as the target
y = df['frp']

#drop frp from the features (X) so the model doesn't cheat
#also drop non-numeric columns for a baseline 
X = df.drop(['frp', 'acq_date', 'satellite', 'instrument', 'daynight', 'confidence'], axis=1)

#split off the 10% Test set
#X_remainder will hold the 90% intended for Train + Validation
X_remainder, X_test_xgb, y_remainder, y_test_xgb = train_test_split(
    X, y, test_size=0.10, random_state=42
)

#split the remainder into 70% Train and 20% Validation
#test_size here is 0.22 because 0.20 / 0.90 is roughly 0.22
X_train, X_val, y_train, y_val = train_test_split(
    X_remainder, y_remainder, test_size=0.22, random_state=42
)

#train with early stopping
#pass the validation set to eval_set so XGBoost can monitor performance
xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False )

#final predictions
xgb_predictions = xgb_model.predict(X_test_xgb)


#calculate metrics
xgb_r2 = r2_score(y_test_xgb, xgb_predictions)
xgb_mse = mean_squared_error(y_test_xgb, xgb_predictions)
xgb_rmse = np.sqrt(xgb_mse)


#gradient boost model
#convert daynight to 1 and 0 so the model can read it
df['daynight'] = df['daynight'].map({'D': 1, 'N': 0})

#drop acq_date and acq_time as dates need special handling
features = ['latitude', 'longitude', 'brightness', 'scan', 'track', 'bright_t31', 'daynight', 'type']
X = df[features]
y = df['frp']

#split into 70% for training 20% for validation 10% 
X_temp, X_test_gb, y_temp, y_test_gb = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.222, random_state=42)

#train the model
gb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, early_stopping_rounds=15)

gb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

#final predictions
gb_predictions = gb_model.predict(X_test_gb)


#calculate metrics
gb_r2 = r2_score(y_test_gb, gb_predictions)
gb_mse = mean_squared_error(y_test_gb, gb_predictions)
gb_rmse = np.sqrt(gb_mse)


#gather everything for CSV file
final_df = pd.DataFrame({
    #Linear Regression columns
    'lr_frp': y_test_lr.values,
    'lr_predict_frp': lr_predictions,
    'lr_frp_difference': y_test_lr.values - lr_predictions,
    'lr_R2': lr_r2,
    'lr_RMSE': lr_rmse,
    
    #XGBoost columns
    'xgb_frp': y_test_xgb.values,
    'xgb_predict_frp': xgb_predictions,
    'xgb_frp_difference': y_test_xgb.values - xgb_predictions,
    'xgb_R2': xgb_r2,
    'xgb_RMSE': xgb_rmse,
    
    #Gradient Boost columns
    'gb_frp': y_test_gb.values,
    'gb_predict_frp': gb_predictions,
    'gb_frp_difference': y_test_gb.values - gb_predictions,
    'gb_R2': gb_r2,
    'gb_RMSE': gb_rmse
})

# Save to CSV
final_df.to_csv('enhanced_baseline_model_results.csv', index=False)

print("finished creating CSV file")