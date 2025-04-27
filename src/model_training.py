# src/modeling.py

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    return xgb
