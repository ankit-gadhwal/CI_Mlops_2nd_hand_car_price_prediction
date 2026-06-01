import tensorflow as tf
import pandas as pd
import numpy as np
import os
import yaml
from tensorflow.keras.layers import (Dense,Normalization,InputLayer)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import (RootMeanSquaredError)

def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data : {e}")
def load_params(filepath):
    try:
        with open(filepath,"r") as file:
            params = yaml.safe_load(file)
        return (params["model_building"]["unit1"],params["model_building"]["unit2"],params["model_building"]["unit3"],params["model_building"]["unit4"],
                params["model_building"]["learning_r"],params["model_building"]["epoch"])
    except Exception as e:
        raise Exception(f"error occured in parameter loading {e}")
def prepare_data(df):
    X = df.drop(
        columns = ["current price"]
    )
    y = df["current price"]
    return X,y

def create_dataset(X,y,batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X.values,y.values))
    dataset = dataset.shuffle(buffer_size = len(X),reshuffle_each_iteration = True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_model(normalizer,unit1,unit2,unit3,unit4,l_r,input_dim):
    model = Sequential([
        InputLayer(shape=(input_dim,)),
        normalizer,
        Dense(135,activation = "relu"),
        Dense(128,activation="relu"),
        Dense(100,activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=l_r),
        loss=MeanAbsoluteError(),
        metrics = [RootMeanSquaredError()
                   ]
    )
    return model

def main():
    try:
        processed_path = "data/processed"
        params_filepath = "params.yaml"
        train_df = load_data(os.path.join(processed_path,"train_processed.csv"))
        val_df = load_data(os.path.join(processed_path,"val_processed.csv"))
        X_train,y_train = prepare_data(train_df)
        X_val,y_val = prepare_data(val_df)
        normalizer = Normalization()
        normalizer.adapt(X_train.values)
        train_dataset = create_dataset(X_train,y_train)
        val_dataset = create_dataset(X_val,y_val)
        (unit1,unit2,unit3,unit4,l_r,epoch) = load_params(params_filepath)
        # Build model
        model = build_model(normalizer,unit1,unit2,unit3,unit4,l_r,X_train.shape[1])

        # Train model

        history = model.fit(train_dataset,validation_data = val_dataset,epochs=epoch,verbose=1)
        
        model.save("car_price_model.h5")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        tflite_model = converter.convert()

        with open("car_price_model.tflite","wb") as f:
            f.write(tflite_model)

    except Exception as e:
        raise Exception(f"Error occured: {e}")

if __name__ == "__main__":
    main()
