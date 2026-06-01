import os
# import tensorflow as tf
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
def load_data(filepath:str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"error occured in data loading {e}")
    
def load_params(filepath:str):
    try:
        with open(filepath,"r") as file:
            params = yaml.safe_load(file)
        return (params["data_collection"]["TRAINRATIO"],params["data_collection"]["VAL_RATIO"],params["data_collection"]["TEST_RATIO"])
    except Exception as e:
        raise Exception(f"Error occur in parameter loading {e}")
    
def split_data(data : pd.DataFrame,train_ratio:float,val_ratio: float,test_ratio : float):
    try:
        train_data,temp_data = train_test_split(data,
                                                test_size=(1-train_ratio),
                                                random_state=42,
                                                shuffle = True)
        # Validation + Test Split
        val_relative_ratio = val_ratio / (val_ratio + test_ratio)
        val_data,test_data = train_test_split(
            temp_data,
            test_size=(1-val_relative_ratio),
            random_state=42,
            shuffle=True
        )
        return (train_data,val_data,test_data)
    except Exception as e:
        raise Exception(f"error occured while spliting data {e}")

def save_data(df : pd.DataFrame,filepath: str) ->None:
    try:
        df.to_csv(filepath,index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath} : {e}")

def main():
    try:
        data_filepath = r"C:\Users\ankit\OneDrive\Desktop\train.csv"
        params_filepath = "params.yaml"
        raw_data_path = os.path.join("data","raw")
        data = load_data(data_filepath)
        (TRAINRATIO,VAL_RATIO,TEST_RATIO) = load_params(params_filepath)
        (train_data,val_data,test_data) = split_data(data,TRAINRATIO,VAL_RATIO,TEST_RATIO)
        os.makedirs(raw_data_path,exist_ok=True)
        save_data(train_data,os.path.join(raw_data_path,"train.csv"))
        save_data(val_data,os.path.join(raw_data_path,"val.csv"))
        save_data(test_data,os.path.join(raw_data_path,"test.csv"))
    except Exception as e:
        raise Exception(f"An error occured : {e}")
    
if __name__ == "__main__":
    main()
