import argparse
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Any, Tuple


XGBOOST_DEFAULT_PARAMS = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'random_state': 42
    }

FOLDER = "./models/"

def train_xgboost(X, y, test_size=0.2, previous_model=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = xgb.XGBRegressor(**XGBOOST_DEFAULT_PARAMS)
    model.fit(X_train, y_train, xgb_model=previous_model)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) 
    r2 = r2_score(y_test, y_pred)

    print(f"Training XGBoost model with error R2={r2} and RMSE={rmse}")
    return model


def initial_training(filename: str,  objs: List[str], model_name="xgboost", save=True) -> None:
    """Function to simulate initial training."""
    print(f"Initial training with file: {filename} with {objs}")

    df = pd.read_csv(filename, index_col=0)
    X = df.drop(columns=objs, axis=1)
    y = df[objs]

    model = train_xgboost(X.values, y.values)
    
    model_output_name = None
    if save:
        # Check if the folder exists, if not create it
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
        filename_ = filename.split("/")[-1] if "/" in filename else filename    
        filename_ = filename_.removesuffix(".csv").replace("_"+CANDIDATES_SELECTION_PREFIX3, "")
        model_output_name = FOLDER + filename_ + "_" + model_name + ".json"   
        model.save_model(model_output_name) # Saves in JSON format
    print(f"<< Surrogates: Initial model training completed - {model_output_name}")


CANDIDATES_SELECTION_PREFIX2 = "marked"
def selecting_candidates(filename: str, fraction: float) -> None:
    """Function to simulate selecting candidates."""
    print(f"Selecting candidates with file: {filename} and fraction={fraction}")
    df = pd.read_csv(filename, index_col=0)
    
    filename_ = filename.split("/")[-1] if "/" in filename else filename
    folder_ = filename.removesuffix(filename_)
    filename_ = filename_.removesuffix(".csv")
    output_name = folder_ + filename_.replace(filename_, filename_+"_"+CANDIDATES_SELECTION_PREFIX2) + ".csv"
    df.to_csv(output_name, index=False)
    print(f">> marked data saved to {output_name}")
    print(f"<< Surrogates: Candidates marked")


CANDIDATES_SELECTION_PREFIX3 = "evaluated"
def retraining(filename: str, objs: List[str], model_name="xgboost", save=True) -> None:
    """Function to simulate retraining."""
    print(f"Retraining with file: {filename} with {objs}")

    df = pd.read_csv(filename, index_col=0)
    X = df.drop(columns=objs, axis=1)
    y = df[objs]

    filename_ = filename.split("/")[-1] if "/" in filename else filename    
    filename_ = filename_.removesuffix(".csv").replace("_"+CANDIDATES_SELECTION_PREFIX3, "")
    model_output_name = FOLDER + filename_ + "_" + model_name + ".json"   
    
    # Check for a previous model
    n_round = int(filename_.split("_")[0])
    previous_round = n_round - 1
    previous_model_output_name = FOLDER + filename_.replace(str(n_round), str(previous_round)) + "_" + model_name + ".json"
    print(f"Round: {n_round} {previous_model_output_name} --> {model_output_name}")
    previous_model = None
    if os.path.exists(previous_model_output_name):
        m = xgb.XGBRegressor()
        m.load_model(previous_model_output_name)
        previous_model = m.get_booster()  # Get the booster from the model
    
    model = train_xgboost(X.values, y.values, previous_model=previous_model)

    if save:
        model.save_model(model_output_name) # Saves in JSON format
    print(f"<< Surrogates: Re-training completed - {model_output_name}")


PREFIX_PREDICTION = "predicted"
def predicting(filename: str, objs: List[str], model_name="xgboost") -> None:
    """Function to simulate predicting."""
    print(f"Predicting with file: {filename}")
    df = pd.read_csv(filename, index_col=0)
  
    filename_ = filename.split("/")[-1] if "/" in filename else filename
    folder_ = filename.removesuffix(filename_)
    model_filename_ = filename_.removesuffix(".csv").replace("_"+CANDIDATES_SELECTION_PREFIX3, "")
    model_output_name = FOLDER + model_filename_ + "_" + model_name + ".json"   
  
    # print(filename_, model_filename_, model_output_name)
    if os.path.exists(model_output_name):
        X_test = df.drop(columns=objs, axis=1)
        y_test = df[objs]

        model = xgb.XGBRegressor()
        model.load_model(model_output_name)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) 
        r2 = r2_score(y_test, y_pred)
        print(f"Testing XGBoost model with error R2={r2} and RMSE={rmse}")
    
    output_name = folder_ + filename_.replace(CANDIDATES_SELECTION_PREFIX3, PREFIX_PREDICTION)
    df.to_csv(output_name, index=False)
    print(f">> predicted data saved to {output_name}")
    print(f"<< Surrogates: Predictions with new model completed")
    

#############################################################


parser = argparse.ArgumentParser(description="Do nothing script for testing purposes.")
parser.add_argument('-i', '--init', type=str, required=False)
parser.add_argument('-s', '--select', type=str, required=False)
parser.add_argument('-o', '--objectives', nargs='+', required=False)
parser.add_argument('-f', '--fraction', type=float, required=False)
parser.add_argument('-r', '--retrain', type=str, required=False)
parser.add_argument('-p', '--predict', type=str, required=False)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Running Python script ...") #, args)

    if args.init and args.objectives:
        initial_training(args.init, args.objectives)

    if args.select and args.fraction:
        selecting_candidates(args.select, args.fraction)

    if args.retrain and args.objectives:
        retraining(args.retrain, args.objectives)

    if args.predict and args.objectives:
        predicting(args.predict, args.objectives)
