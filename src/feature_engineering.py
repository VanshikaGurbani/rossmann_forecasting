# src/feature_engineering.py

import pandas as pd
import numpy as np

def create_lag_features(df, lags=[7, 14, 30]):
    for lag in lags:
        df[f'Sales_Lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)
    return df

def create_rolling_features(df, windows=[7, 30]):
    for window in windows:
        df[f'Sales_RollingMean_{window}'] = df.groupby('Store')['Sales'].shift(1).rolling(window=window).mean()
    return df

def create_promo_features(df):
    df['Promo_Duration'] = df.groupby('Store')['Promo'].cumsum()
    return df

def create_competition_features(df):
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0).astype(int)
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0).astype(int)
    df['CompetitionOpenDate'] = pd.to_datetime(dict(year=df['CompetitionOpenSinceYear'],
                                                     month=df['CompetitionOpenSinceMonth'],
                                                     day=15), errors='coerce')
    def calculate_month_diff(current, past):
        if pd.isnull(past):
            return 0
        return (current.year - past.year) * 12 + (current.month - past.month)

    df['CompetitionOpenTimeMonths'] = df.apply(lambda row: calculate_month_diff(row['Date'], row['CompetitionOpenDate']), axis=1)
    df['CompetitionOpenTimeMonths'] = df['CompetitionOpenTimeMonths'].clip(lower=0)
    return df
