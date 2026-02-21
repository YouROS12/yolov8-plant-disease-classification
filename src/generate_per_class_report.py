# ==========================================
# Generate Per-Class Performance Report (Colab Cell)
# ==========================================
# Instructions:
# 1. Copy this entire cell into your Google Colab notebook.
# 2. Run it. It will load your trained model and test data.
# 3. It prints the detailed per-class classification report.
# 4. It saves 'per_class_report.csv' in your Drive folder.
# ==========================================

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from google.colab import drive

# 1. Mount Drive
drive.mount('/content/drive')

# --- CONFIGURATION (Match your settings) ---
BASE_PATH = '/content/drive/MyDrive/reviewers_suggestion_paper1/'
BACKBONE = 'yolo'
COMPRESSION = 'ipca'
LABEL_COUNT = 11  # Treatment-based grouping

# Construct Filenames
suffix = f"_{BACKBONE}_{COMPRESSION}"
feature_file = os.path.join(BASE_PATH, f'x_test{suffix}.npy')
label_file = os.path.join(BASE_PATH, f'y_test_{LABEL_COUNT}{suffix}.npy')
model_file = os.path.join(BASE_PATH, f'svm_model_{BACKBONE}_{COMPRESSION}_{LABEL_COUNT} Classes.joblib')
encoder_file = os.path.join(BASE_PATH, f'encoder_{LABEL_COUNT}{suffix}.joblib')

print(f"Looking for files in: {BASE_PATH}")
print(f"Model: {os.path.basename(model_file)}")

# 2. Load Data & Model
if not os.path.exists(model_file):
    print(f"[ERROR] Model file not found! Check path: {model_file}")
else:
    print("Loading data...")
    x_test = np.load(feature_file)
    y_test = np.load(label_file)
    
    print("Loading model...")
    model = joblib.load(model_file)
    
    # Try to load encoder for class names
    if os.path.exists(encoder_file):
        encoder = joblib.load(encoder_file)
        target_names = encoder.classes_
        print(f"Loaded encoder with classes: {target_names}")
    else:
        print("[WARNING] Encoder not found. Using generic class IDs.")
        target_names = [str(i) for i in range(len(np.unique(y_test)))]

    # 3. Predict
    print("Running inference on test set...")
    y_pred = model.predict(x_test)

    # 4. Generate Report
    acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> TEST ACCURACY: {acc:.4f} ({acc*100:.2f}%) <<<\n")

    print("--- Per-Class Classification Report ---")
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 5. Save to CSV (for easy copy-paste to Table 4)
    df_report = pd.DataFrame(report_dict).transpose()
    save_path = os.path.join(BASE_PATH, 'per_class_report_NEW.csv')
    df_report.to_csv(save_path)
    print(f"\n[SUCCESS] Report saved to: {save_path}")
    print("You can verify the values in this CSV for Table 4.")
