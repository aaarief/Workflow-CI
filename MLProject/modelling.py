import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import dagshub
import os
import json
import shutil
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.utils import estimator_html_repr
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI ---
DAGSHUB_REPO_OWNER = "aaarief" 
DAGSHUB_REPO_NAME = "membangun-model" # Sesuaikan dengan nama repo Anda

def load_data():
    # Path relatif terhadap folder MLProject ini
    # Asumsi: folder bank_marketing_preprocessing ada di sebelah script ini
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "bank_marketing_preprocessing")
    
    train_path = os.path.join(data_dir, "train_processed.csv")
    test_path = os.path.join(data_dir, "test_processed.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Data tidak ditemukan di: {train_path}")
        
    return pd.read_csv(train_path), pd.read_csv(test_path)

def train():
    print("Starting training pipeline...")
    
    # Setup Tracking
    # Jika env var MLFLOW_TRACKING_URI belum ada (misal run lokal), kita init manual
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    
    mlflow.set_experiment("Bank Marketing CI/CD Pipeline")
    
    train_df, test_df = load_data()
    
    X_train = train_df.drop(columns=['y'])
    y_train = train_df['y']
    X_test = test_df.drop(columns=['y'])
    y_test = test_df['y']
    
    with mlflow.start_run() as run:
        # Hyperparameter tuning (Sederhana agar cepat di CI)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, None]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Log Params
        mlflow.log_params(best_params)
        
        # Predict & Metrics
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc_val = auc(*roc_curve(y_test, y_prob)[:2])
        
        metrics = {
            "accuracy": acc, 
            "f1_score": f1, 
            "roc_auc": roc_auc_val
        }
        mlflow.log_metrics(metrics)
        print(f"Model Accuracy: {acc}")

        # --- LOG MODEL (PENTING UNTUK DOCKER) ---
        # Kita beri nama registered_model_name agar mudah dipanggil saat build docker
        model_name = "BankMarketingModel_CI"
        
        print("Logging model...")
        
        # Signature & Input Example
        signature = infer_signature(X_train, y_pred)
        input_example = X_train.iloc[:5]

        try:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                registered_model_name=model_name,
                signature=signature,
                input_example=input_example
            )
            print("✅ Model logged and registered successfully!")
        except Exception as e:
            print(f"❌ Failed to log model directly: {e}")
            # Fallback logic if needed, but for CI/CD we usually want it to fail if model logging fails
            raise e
        
        # --- ARTEFAK VISUALISASI ---
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        # 2. Metric Info JSON
        with open("metric_info.json", "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact("metric_info.json")

        print(f"Run ID: {run.info.run_id}")
        print("Training finished.")

if __name__ == "__main__":
    train()
