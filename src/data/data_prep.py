import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_data(filepath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"error occured in data loading {e}")
    
def feature_engineering(df):
     
     df["depreciation"] = (
        df["on road old"] -
        df["current price"]
    )

    # km per year
     df["km_per_year"] = (
        df["km"] / (df["years"] + 1)
    )

    # performance score
     df["performance"] = (
        df["hp"] + df["torque"]
       )

    # value score
     df["value_score"] = (
        df["economy"] * df["rating"]
     )

     return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if "v.id" in df.columns:
        df = df.drop(columns=["v.id"])
    
    # Store columns before transform
    
    # Feature Engineering
    df = feature_engineering(df)
    cols = df.columns
    # Numeric Columns
    # num_cols = df.select_dtypes(
    #     include=[np.number]
    # ).columns
    
    # Handele Missing Values
    imputer = SimpleImputer(
        strategy="median"
    )

    df = imputer.fit_transform(df)
    df = pd.DataFrame(df,columns=cols)
    return df

# def normalize_data(train_df,val_df,test_df):
#     scaler = StandardScaler()
#     feature_cols = [col for col in train_df.columns
#                     if col != "current price"]
    
#     # Fit Only on train
#     train_df[feature_cols] = scaler.fit_transform(
#         train_df[feature_cols]
#     )

#     # Transform validation
#     val_df[feature_cols] =  scaler.fit_transform(
#         val_df[feature_cols]
#     )

#     # Transform test
#     test_df[feature_cols] = scaler.fit_transform(
#         test_df[feature_cols]
#     )

#     return (
#         train_df,
#         val_df,
#         test_df
    # )

def save_data(df,filepath):
    try:
        df.to_csv(filepath,index=False)
    except Exception as e:
        raise Exception(f"Error saving data: {e}")
    
def main():
    try:
        raw_data_path = "data/raw"
        processed_data_path = "data/processed"
        train_df = load_data(os.path.join(raw_data_path, "train.csv"))
        val_df = load_data(os.path.join(raw_data_path, "val.csv"))
        test_df = load_data(os.path.join(raw_data_path, "test.csv"))
        train_df = preprocess(train_df)
        val_df = preprocess(val_df)
        test_df = preprocess(test_df)
        # (Ntrain_df,Nval_df,Ntest_df) = normalize_data(train_df,val_df,test_df)
        os.makedirs(processed_data_path,exist_ok=True)
        save_data(train_df,os.path.join(processed_data_path,"train_processed.csv"))
        save_data(val_df,os.path.join(processed_data_path,"val_processed.csv"))
        save_data(test_df,os.path.join(processed_data_path,"test_processed.csv"))
    except Exception as e:
        raise Exception(f"error occured in main function {e}")
    
if __name__ == "__main__":
    main()