import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Catatan: Tidak perlu import dagshub atau set_experiment.
# Kredensial dan URI sudah diatur lewat Environment Variables di GitHub Actions.

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "bank_marketing_preprocessing")
    train_path = os.path.join(data_dir, "train_processed.csv")
    test_path = os.path.join(data_dir, "test_processed.csv")
    return pd.read_csv(train_path), pd.read_csv(test_path)

def train():
    print("Starting training pipeline (CI/CD)...")
    
    # Load Data
    train_df, test_df = load_data()
    X_train = train_df.drop(columns=['y'])
    y_train = train_df['y']
    X_test = test_df.drop(columns=['y'])
    y_test = test_df['y']
    
    # Enable Autolog (untuk metrics otomatis)
    mlflow.sklearn.autolog()
    
    run_id = None
    
    # Start Run
    # Kita tidak set_experiment di sini agar mengikuti konteks dari 'mlflow run'
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Run ID: {run_id}")
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        
        # EXPLICIT LOGGING (WAJIB UNTUK DOCKER)
        # Kita simpan model secara eksplisit ke folder "model"
        # Ini memastikan path 'runs:/.../model' nanti valid
        mlflow.sklearn.log_model(model, "model")
        print("✅ Model logged explicitly to 'model' artifact path.")
        
    # Register Model (Dilakukan SETELAH run selesai agar artifact pasti sudah ter-upload)
    if run_id:
        model_uri = f"runs:/{run_id}/model"
        print(f"Registering model from URI: {model_uri} ...")
        try:
            mlflow.register_model(model_uri, "BankMarketingModel_CI")
            print("✅ Model registered successfully as 'BankMarketingModel_CI'")
        except Exception as e:
            print(f"❌ Failed to register model: {e}")
            # Jangan raise error agar pipeline tidak merah jika hanya gagal register (opsional)
            # raise e 

if __name__ == "__main__":
    train()
