import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
import yaml
from sklearn.pipeline import Pipeline

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")
from src.logger import logging


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise


# Suppress MLflow artifact download warnings
# os.environ["MLFLOW_DISABLE_ARTIFACTS_DOWNLOAD"] = "1"

# Set MLflow Tracking URI & DAGsHub integration
MLFLOW_TRACKING_URI = "https://dagshub.com/alysahab/Mlops-Capstone-Project.mlflow"
dagshub.init(repo_owner="alysahab", repo_name="Mlops-Capstone-Project", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("LoR Hyperparameter Tuning")


# ==========================
# Text Preprocessing Functions
# ==========================
def preprocess_text(text):
    """Applies multiple text preprocessing steps."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # Remove punctuation
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatization & stopwords removal
    
    return text.strip()


# ==========================
# Load & Prepare Data
# ==========================
def load_and_prepare_data(filepath):
    """Loads, preprocesses, and vectorizes the dataset."""
    df = pd.read_csv(filepath)
    return df


def prepare_data(data: pd.DataFrame):
    """Prepares the data for model training and testing."""
    # load parameter
    params = load_params("params.yaml")
    
    # Apply text preprocessing
    data["review"] = data["review"].astype(str).apply(preprocess_text)
    
    # Filter for binary classification
    data = data[data["sentiment"].isin(["positive", "negative"])]
    data["sentiment"] = data["sentiment"].map({"negative": 0, "positive": 1})
    
    X = data["review"]
    y = data["sentiment"]

    return X,y

    
    

# ==========================
# Train & Log Model
# ==========================
def train_and_log_model(X, y):
    """Trains the Logistic Regression model and logs it to MLflow."""
    
    params = load_params("params.yaml")

    with mlflow.start_run(run_name="Logistic_Regression_Full_Data") as run:

        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=params["feature_engineering"]["max_features"],
            )),
            ('model', LogisticRegression(
                solver=params["model_training"]["solver"],
                penalty=params["model_training"]["penalty"],
                C=params["model_training"]["C"],
                class_weight=params["model_training"]["class_weight"],
                max_iter=params["model_training"]["max_iter"],
                random_state=params["model_training"]["random_state"]
            ))
        ])

        # Define multiple metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'auc': 'roc_auc'
        }

        # Cross-validation
        cv_results = cross_validate(
            pipeline,
            X, y,
            cv=5,
            scoring=scoring,
            return_train_score=False
        )

        # Log each fold metric + mean
        for metric in scoring.keys():
            fold_scores = cv_results[f'test_{metric}']

            # Log mean and std
            mlflow.log_metric(f"{metric}_mean", fold_scores.mean())
            mlflow.log_metric(f"{metric}_std", fold_scores.std())

            # Log individual fold results
            # for i, score in enumerate(fold_scores):
            #     mlflow.log_metric(f"{metric}_fold_{i+1}", score)

        # Log model parameters for reproducibility
        mlflow.log_params({
            "solver": params["model_training"]["solver"],
            "penalty": params["model_training"]["penalty"],
            "C": params["model_training"]["C"],
            "class_weight": params["model_training"]["class_weight"],
            "max_iter": params["model_training"]["max_iter"],
            "random_state": params["model_training"]["random_state"],
            "max_features": params["feature_engineering"]["max_features"]
        })

        # Fit final model on all data and log it
        pipeline.fit(X, y)
        mlflow.sklearn.log_model(pipeline, "logistic_regression_model")

        logging.info("Model and metrics logged successfully to MLflow.")

        
        


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    data = load_and_prepare_data("notebooks/IMDB.csv")
    X,y = prepare_data(data)
    train_and_log_model(X,y)

    
