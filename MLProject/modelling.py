import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import dagshub

# --- KONFIGURASI ---
DAGSHUB_REPO_OWNER = "aaarief" 
DAGSHUB_REPO_NAME = "membangun-model"

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "bank_marketing_preprocessing")
    train_path = os.path.join(data_dir, "train_processed.csv")
    test_path = os.path.join(data_dir, "test_processed.csv")
    return pd.read_csv(train_path), pd.read_csv(test_path)

def train():
    print("Starting training pipeline (CI/CD)...")
    
    # Setup Tracking (Penting untuk CI/CD agar lapor ke DAGsHub)
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    
    mlflow.set_experiment("Bank Marketing CI/CD Pipeline")
    
    train_df, test_df = load_data()
    X_train = train_df.drop(columns=['y'])
    y_train = train_df['y']
    X_test = test_df.drop(columns=['y'])
    y_test = test_df['y']
    
    # Enable Autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        
        # Register Model untuk keperluan Docker
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "BankMarketingModel_CI")
        print("✅ Model registered as 'BankMarketingModel_CI'")

if __name__ == "__main__":
    train()
