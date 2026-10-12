# src/causal_impact_manual.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def run_manual_causal_impact(df, pre_period, post_period, features, target_col='Sales'):
    pre_data = df[(df['Date'] >= pre_period[0]) & (df['Date'] <= pre_period[1])]
    post_data = df[(df['Date'] >= post_period[0]) & (df['Date'] <= post_period[1])]

    X_pre = pre_data[features]
    y_pre = pre_data[target_col]
    X_post = post_data[features]
    y_post = post_data[target_col]

    model = LinearRegression()
    model.fit(X_pre, y_pre)
    y_pred_cf = model.predict(X_post)

    lift = y_post.values - y_pred_cf
    avg_lift = np.mean(lift)
    rel_lift = (avg_lift / np.mean(y_pred_cf)) * 100

    return avg_lift, rel_lift, y_post, y_pred_cf
