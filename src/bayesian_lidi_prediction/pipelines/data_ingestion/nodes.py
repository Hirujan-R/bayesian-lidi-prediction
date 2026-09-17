import pandas as pd
import os

def convert_csv_to_parquet(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError
    try:
        df = pd.read_csv(filepath)
    except:
        raise TypeError("File path must lead to a csv file.")
    
    return df