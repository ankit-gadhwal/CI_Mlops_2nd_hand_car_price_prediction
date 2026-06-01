import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
from tensorflow.keras.models import load_model
# import tflite_runtime.interpreter as tflite
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
# from dvclive import Live
import yaml
import mlflow.tensorflow
import dagshub
import mlflow
from mlflow.models import infer_signature

dagshub.init(repo_owner='ankit-gadhwal', repo_name='CI_Mlops_2nd_hand_car_price_prediction', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/ankit-gadhwal/CI_Mlops_2nd_hand_car_price_prediction.mlflow")

def get_or_create_experiment_id(experiment_name):

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"New experiment created: {experiment_id}")
        return experiment_id
    print(f"Existing experiment found: "f"{experiment.experiment_id}")
    return experiment.experiment_id

def data_load(filepath):
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"error occured in data loading {e}")
    
def data_prepare(df):
    try:
        X_test = df.drop(columns = "current price")
        y_test = df["current price"]
        return X_test,y_test
    except Exception as e:
        raise Exception(f"error occured in data preparation {e}")
    
def save_metrics(metrics,filepath):
    try:
        with open(filepath,"w") as file:
            json.dump(metrics,file,indent=4)
            
    except Exception as e:
        raise Exception(f"Error saving metrics: {e}")
    
def main():
    try:
        processed_path = os.path.join("data","processed")
        reports_path = "reports"
        os.makedirs(
            reports_path,
            exist_ok=True
        )

        # load model
        model = load_model("car_price_model.h5")
        
        # load test data
        test_df = data_load(os.path.join(processed_path,"test_processed.csv"))

        # prepare data
        X_test,y_test = data_prepare(test_df)

        # Evaluate model
        loss,rmse = model.evaluate(
            X_test.values,
            y_test.values,
            verbose=0
        )

        # predictions
        y_pred = model.predict(X_test.values)
        y_pred = y_pred.flatten()
        
        mae = float(mean_absolute_error(y_test,y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test,y_pred)))
        r2score = float(r2_score(y_test,y_pred))
        with open("params.yaml","r") as file:
            params = yaml.safe_load(file)
        (train_ratio,val_ratio,test_ratio) = (params["data_collection"]["TRAINRATIO"],params["data_collection"]["VAL_RATIO"],params["data_collection"]["TEST_RATIO"])
        (unit1,unit2,unit3,unit4,l_r,epoch) = (params["model_building"]["unit1"],params["model_building"]["unit2"],params["model_building"]["unit3"],params["model_building"]["unit4"],params["model_building"]["learning_r"],params["model_building"]["epoch"])
        # with Live(save_dvc_exp = True) as live:
        #     live.log_metric("mae",mae)
        #     live.log_metric("rmse",rmse)
        #     live.log_metric("r2_score",r2score)
        #     live.log_param("unit1",unit1)
        #     live.log_param("unit1",unit2)
        #     live.log_param("unit1",unit3)
        #     live.log_param("unit1",unit4)
        experimentId = get_or_create_experiment_id("Car Price Prediction_dvc")
        with mlflow.start_run(experiment_id = experimentId) as run:
            mlflow.log_param("dense_layer_1_neurons",unit1)
            mlflow.log_param("dense_layer_2_neurons",unit2)
            mlflow.log_param("dense_layer_3_neurons",unit3)
            mlflow.log_param("dense_layer_4_neurons",unit4)
            mlflow.log_param("learning_rate",l_r)
            mlflow.log_param("epochs",epoch)
            mlflow.log_param("batch_size",32)
            # Metrics
            metrics = {
                "mae": mae,
                "rmse": rmse,
                "r2score": r2score,
            }
        
            # save metrics
            save_metrics(metrics,os.path.join(reports_path,"metrics.json"))
            signature = infer_signature(X_test.to_numpy(),model.predict(X_test.to_numpy()))
            mlflow.log_metric("mae",mae)
            mlflow.log_metric("rmse",rmse)
            mlflow.log_metric("r2_score",r2score)
            mlflow.tensorflow.log_model(model,artifact_path="model",signature = signature)
            
            # mlflow.log_artifact(__file__)
            print("RUN ID =", run.info.run_id)
            run_info = {'run_id':run.info.run_id,'model_name': "model"}
            reports_path = "reports/run_info.json"
            with open(reports_path,"w") as file:
                json.dump(run_info,file,indent=4)
    except Exception as e:
        raise Exception(f"Error occured: {e}")   
    
if __name__ == "__main__":
    main()