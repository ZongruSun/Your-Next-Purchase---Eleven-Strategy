import pandas as pd

def load_data():
    df1 = pd.read_csv("data/merged_data_with_features.csv")
    df_stocks = pd.read_csv("data/stocks.csv")
    return df1, df_stocks
