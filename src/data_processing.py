# src/data_processing.py

import pandas as pd

def load_data(train_path, store_path):
    train = pd.read_csv(train_path)
    store = pd.read_csv(store_path)
    df = pd.merge(train, store, on='Store', how='left')
    return df

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df = df[df['Open'] == 1]
    df = df[df['Sales'] > 0]
    return df
