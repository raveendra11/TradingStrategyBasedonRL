import pandas as pd

def load_stock_data(filepath):

    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    return df, data
