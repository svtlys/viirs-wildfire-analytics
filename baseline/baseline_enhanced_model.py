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
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error
import joblib

df = pd.read_csv('stratified_sample.csv')

#linear regression model
#features and target for prediction
features = ['latitude', 'longitude', 'brightness', 'scan', 'track', 'acq_date', 'acq_time', 'satellite', 'instrument', 'confidence', 'version', 'bright_t31', 'daynight', 'type']
target = 'frp'

X = df[features].copy()
y = df[target]

#encode non-numerical columns
X['acq_date'] = pd.to_datetime(X['acq_date'])
X['acq_date'] = (X['acq_date'] - X['acq_date'].min()).dt.days

X['daynight'] = X['daynight'].map({'D': 1, 'N': 0})
X = pd.get_dummies(X, columns=['satellite', 'instrument', 'type', 'version', 'confidence'], drop_first=True)

X['acq_time'] = pd.to_numeric(X['acq_time'], errors='coerce').fillna(0)

#split 20% for testing and 80% for training 
X_train, X_test_lr, y_train, y_test_lr = train_test_split(X, y, test_size=0.2, random_state=42)

#create the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

joblib.dump(lr_model, 'lr_model.joblib')

#predictions made for testing
lr_predictions = lr_model.predict(X_test_lr)

#calculate metrics
lr_r2 = r2_score(y_test_lr, lr_predictions)
lr_mse = mean_squared_error(y_test_lr, lr_predictions)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test_lr, lr_predictions)


#xgboost model
#set frp as the target
y = df['frp']

#drop frp from the features (X) so the model doesn't cheat
#also drop non-numeric columns for a baseline 
features = ['latitude', 'longitude', 'brightness', 'scan', 'track', 'acq_date', 'acq_time', 'satellite', 'instrument', 'confidence', 'version', 'bright_t31', 'daynight', 'type']
X = df[features].copy()

#encode non-numerical columns
X['acq_date'] = pd.to_datetime(X['acq_date'])
X['acq_date'] = (X['acq_date'] - X['acq_date'].min()).dt.days

X['acq_time'] = pd.to_numeric(X['acq_time'], errors='coerce').fillna(0)

for col in ['satellite', 'instrument', 'daynight', 'type', 'version', 'confidence']:
    X[col] = X[col].astype('category').cat.codes

#split off 20% for testing
X_train_full, X_test_xgb, y_train_full, y_test_xgb = train_test_split(X, y, test_size=0.20, random_state=42)

#split the remainder into 80% for training
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.10, random_state=42)

#train with early stopping
#pass the validation set to eval_set so XGBoost can monitor performance
xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=50)

xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False )

joblib.dump(xgb_model, 'xgb_model.joblib')

#final predictions
xgb_predictions = xgb_model.predict(X_test_xgb)


#calculate metrics
xgb_r2 = r2_score(y_test_xgb, xgb_predictions)
xgb_mse = mean_squared_error(y_test_xgb, xgb_predictions)
xgb_rmse = np.sqrt(xgb_mse)
xgb_mae = mean_absolute_error(y_test_xgb, xgb_predictions)


#gradient boost model

#drop acq_date and acq_time as dates need special handling
features = ['latitude', 'longitude', 'brightness', 'scan', 'track', 'acq_date', 'acq_time', 'satellite', 'instrument', 'confidence', 'version', 'bright_t31', 'daynight', 'type']
X = df[features].copy()
y = df['frp']

#encode non-numerical columns
X['acq_date'] = pd.to_datetime(X['acq_date'])
X['acq_date'] = (X['acq_date'] - X['acq_date'].min()).dt.days

X['acq_time'] = pd.to_numeric(X['acq_time'], errors='coerce').fillna(0)

for col in X.select_dtypes(include=['object', 'string']).columns:
    X[col] = X[col].astype('category').cat.codes

#split off 20% for testing
X_train_full, X_test_gb, y_train_full, y_test_gb = train_test_split(X, y, test_size=0.20, random_state=42)

#split the remainder into 80% for training
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.10, random_state=42)

#train the model
gb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, early_stopping_rounds=15)

gb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

joblib.dump(gb_model, 'gb_model.joblib')

#final predictions
gb_predictions = gb_model.predict(X_test_gb)


#calculate metrics
gb_r2 = r2_score(y_test_gb, gb_predictions)
gb_mse = mean_squared_error(y_test_gb, gb_predictions)
gb_rmse = np.sqrt(gb_mse)
gb_mae = mean_absolute_error(y_test_gb, gb_predictions)


#gather everything for CSV file
final_df = pd.DataFrame({
    #Linear Regression columns
    'lr_frp': y_test_lr.values,
    'lr_predict_frp': lr_predictions,
    'lr_frp_difference': y_test_lr.values - lr_predictions,
    'lr_R2': lr_r2,
    'lr_RMSE': lr_rmse,
    'lr_MAE': lr_mae,
    
    #XGBoost columns
    'xgb_frp': y_test_xgb.values,
    'xgb_predict_frp': xgb_predictions,
    'xgb_frp_difference': y_test_xgb.values - xgb_predictions,
    'xgb_R2': xgb_r2,
    'xgb_RMSE': xgb_rmse,
    'xgb_MAE': xgb_mae,
    
    #Gradient Boost columns
    'gb_frp': y_test_gb.values,
    'gb_predict_frp': gb_predictions,
    'gb_frp_difference': y_test_gb.values - gb_predictions,
    'gb_R2': gb_r2,
    'gb_RMSE': gb_rmse,
    'gb_MAE': gb_mae
})

# Save to CSV
final_df.to_csv('enhanced_baseline_model_results.csv', index=False)

print("finished creating CSV file, all models are saved")