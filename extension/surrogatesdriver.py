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

def train_xgboost(X, y, test_size=0.2, stratify=None, previous_model=None):

    s = None
    if stratify is not None:
        print(f"Applying stratification on level!")
        s = X[stratify]
        X.drop(stratify, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=s, random_state=42)

    model = xgb.XGBRegressor(**XGBOOST_DEFAULT_PARAMS)
    model.fit(X_train, y_train, xgb_model=previous_model)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) 
    r2 = r2_score(y_test, y_pred)

    print(f"Training XGBoost model - Test error R2={r2} and RMSE={rmse}")
    return model


def initial_training(filename: str,  objs: List[str], model_name="xgboost", save=True) -> None:
    """Function to simulate initial training."""
    print(f"Initial training with file: {filename} with {objs}")

    df = pd.read_csv(filename)
    X = df.drop(columns=objs, axis=1)
    y = df[objs]

    if 'all' in filename:
        model = train_xgboost(X, y, stratify='level')
    else:
        model = train_xgboost(X, y)
    
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
def selecting_candidates(filename: str, fraction: float, objs: List[str]) -> None:
    """Function to simulate selecting candidates."""
    print(f"Selecting candidates with file: {filename}, fraction={fraction} and objectives {objs}")
    df = pd.read_csv(filename)
    # Mark candidates (rows) that need to be evaluated
    # Randomly select k rows and assign their objective values to True, for the remaining objective values assigned them to NaN
    k = round(len(df) * fraction)
    if fraction < 1.0:
        candidates = df.sample(n=k, random_state=42)
    else:
        candidates = df
    df[objs] = np.nan  # Initialize all objective columns to NaN
    # df.loc[~df.index.isin(candidates.index), objs] = np.nan # Set non-candidates to NaN
    df.loc[candidates.index, objs] = 1.0  # Mark selected candidates with True
    print(f"Candidates selected: {len(candidates)} out of {len(df)}")

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

    df = pd.read_csv(filename)
    # Remove first all rows with NaN values
    df = df.dropna()
    print(f"Retraining with {len(df)} samples")
    X = df.drop(columns=objs, axis=1)
    y = df[objs]
    # TODO: It needs to ony use the (marked) values that were evaluated

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
    
    model = train_xgboost(X, y, previous_model=previous_model)

    if save:
        model.save_model(model_output_name) # Saves in JSON format
    print(f"<< Surrogates: Re-training completed - {model_output_name}")


PREFIX_PREDICTION = "predicted"
def predicting(filename: str, objs: List[str], model_name="xgboost") -> None:
    """Function to simulate predicting."""
    print(f"Predicting with file: {filename}")
    df = pd.read_csv(filename)
    # Select the rows with and without marked candidates
    not_nan_df = df[df[objs].notna().any(axis=1)]
    nan_df = df[df[objs].isna().all(axis=1)]

    filename_ = filename.split("/")[-1] if "/" in filename else filename
    folder_ = filename.removesuffix(filename_)
    model_filename_ = filename_.removesuffix(".csv").replace("_"+CANDIDATES_SELECTION_PREFIX3, "")
    model_output_name = FOLDER + model_filename_ + "_" + model_name + ".json"   
  
    # print(filename_, model_filename_, model_output_name)
    if os.path.exists(model_output_name):
        X_train = not_nan_df.drop(columns=objs, axis=1)
        y_train = not_nan_df[objs]
        # It needs to assign the prediction for the non-marked values

        model = xgb.XGBRegressor()
        model.load_model(model_output_name)
        
        y_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        rmse = np.sqrt(mse) 
        r2 = r2_score(y_train, y_pred)
        print(f"Training XGBoost model (increment) - Train error R2={r2} and RMSE={rmse}")

        # Predict for the NaN values
        if len(nan_df) > 0:
            X_pred = nan_df.drop(columns=objs, axis=1)
            y_pred = model.predict(X_pred)
            df.loc[nan_df.index, objs] = y_pred
            # With the available data, there's no way to compute the test error here

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

    if args.select and args.fraction and args.objectives:
        selecting_candidates(args.select, args.fraction, args.objectives)

    if args.retrain and args.objectives:
        retraining(args.retrain, args.objectives)

    if args.predict and args.objectives:
        predicting(args.predict, args.objectives)
