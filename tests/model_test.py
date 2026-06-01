import unittest
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
import os
import pandas as pd
import numpy as np

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "ankit-gadhwal"
repo_name='CI_Mlops_2nd_hand_car_price_prediction'
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

model_name = "model"
class TestModelLoading(unittest.TestCase):
    """unit test class to verify MLflow model loading from the Staging stage"""
    
    def test_model_in_staging(self):
        """Test if the model exists in the Staging Stage"""

        # Initialize the Mlflow client to interact with Mlflow server
        client = MlflowClient()

        # Retrive the latest versions of the models in the 'Staging' stage
        versions = client.get_latest_versions(model_name,stages=["Staging"])

        # Assert that at least one version of the model exists in the 'Staging' stage.
        # If no versions are found,it will raise an error 
        self.assertGreater(len(versions),0,"No model found in the 'Staging' stage")
    def test_model_loading(self):
        """Test if the model can be loaded properly from the Staging stage."""

        # Initialize the Mlflow client again to interact with the server
        client = MlflowClient() 
        # Retrive the latest versions of the models in the 'Staging' stage
        versions = client.get_latest_versions(model_name,stages=["Staging"])

        # If no versions are found,fails the test and skip the modal loading part
        if not versions:
            self.fail("No model found in the 'Staging' stage,skipping model loading test.")

            # get the version details of the latest model in the 'Staging' stage
            latest_version = versions[0].version
            run_id = versions[0].run_id

            # construct the String needed to load the model using its run id
            logged_model = f"runs:/{run_id}/{model_name}"

            try:
                # try to load the model from the specified path
                loaded_model = mlflow.pyfunc.load_model(logged_model)
            except Exception as e:
                # If loading the modals fails,fail the test and output the error message
                self.fail(f"Failed to load the model:{e}")

            self.assertIsNotNone(loaded_model, "The loaded model is None")
            print(f"Model successfully loaded from {logged_model}")
    def test_model_performance(self):
        """Test the performance of the model on test data."""
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Staging"])

        if not versions:
            self.fail("No model found in the 'Staging' stage, skipping performance test.")

        latest_version = versions[0].run_id
        logged_model = f"runs:/{latest_version}/{model_name}"
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        # Load test data
        test_data_path = "data/processed/test_processed.csv"
        if not os.path.exists(test_data_path):
            self.fail(f"Test data not found at {test_data_path}")

        test_data = pd.read_csv(test_data_path)
        X_test = test_data.drop(columns=["current price"])
        y_test = test_data["current price"]

        # Make predictions and calculate metrics
        predictions = loaded_model.predict(X_test)

        mae = float(mean_absolute_error(y_test,predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test,predictions)))
        r2score = float(r2_score(y_test,predictions))

        print(f"mae: {mae}")
        print(f"rmse: {rmse}")
        print(f"r2score: {r2score}")

    # Assert performance metrics meet thresholds
        self.assertLessEqual(mae, 20000, "MAE is above threshold.")
        self.assertLessEqual(rmse, 22000, "RMSE is above threshold.")
        self.assertGreaterEqual(r2score, 0.8, "R2 Score is below threshold.")

if __name__ == "__main__":
    unittest.main()