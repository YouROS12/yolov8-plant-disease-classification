
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# CONFIG
BASE_PATH = '/content/drive/MyDrive/reviewers_suggestion_paper1/'
RESULTS_FILE = os.path.join(BASE_PATH, 'intermediate_results.csv')
OUTPUT_DIR = os.path.join(BASE_PATH, 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. LOAD RESULTS
if not os.path.exists(RESULTS_FILE):
    print(f"ERROR: Results file not found at {RESULTS_FILE}")
    exit(1)

df = pd.read_csv(RESULTS_FILE)
df = df.drop_duplicates(subset=['Backbone', 'Compression', 'Label_Set'], keep='last')
print("Loaded Results (after deduplication):")
print(df)

# 2. PLOT COMPARISON (Bar Chart)
def plot_accuracy_comparison(df):
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Create the bar plot
    ax = sns.barplot(
        data=df, 
        x='Backbone', 
        y='Accuracy', 
        hue='Compression', 
        palette='viridis',
        errorbar=None
    )
    
    # Add labels and title
    plt.title('Method Comparison: Feature Extraction + Compression', fontsize=16)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xlabel('Backbone Model (Frozen Features)', fontsize=12)
    plt.ylim(0.0, 1.0)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'model_comparison_accuracy.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved comparison plot to {save_path}")
    plt.close()

# 3. CONFUSION MATRIX (For Best Model: YOLO IPCA)
def plot_confusion_matrix():
    print("\nGenerating Confusion Matrix for YOLO + IPCA (11 Classes)...")

    # Define best configuration
    best_backbone = 'yolo'
    best_compression = 'ipca'
    best_label_set = 11

    suffix = f"_{best_backbone}_{best_compression}"
    base_file_name_train = f'x_train{suffix}.npy'
    feature_file = os.path.join(BASE_PATH, base_file_name_train)
    label_train_file = os.path.join(BASE_PATH, f'y_train_{best_label_set}{suffix}.npy')
    label_test_file = os.path.join(BASE_PATH, f'y_test_{best_label_set}{suffix}.npy')
    feature_test_file = os.path.join(BASE_PATH, f'x_test{suffix}.npy')
    
    # Model Filename
    model_filename = f"svm_model_{best_backbone}_{best_compression}_{best_label_set} Classes.joblib"

    if not os.path.exists(feature_file) or not os.path.exists(label_train_file):
        print(f"[ERROR] Features/Labels not found for confusion matrix ({base_file_name_train}). Skipping CM.")
        return

    try:
        # Load Data
        print("Loading data for confusion matrix...")
        X_train = np.load(feature_file)
        y_train = np.load(label_train_file)
        X_test = np.load(feature_test_file)
        y_test = np.load(label_test_file)
        
        # Load Saved Model
        print(f"Loading best model for visualization: {model_filename}...")
        model_path = os.path.join(BASE_PATH, model_filename)

        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_filename}")
            print("Please run training_and_evaluation.py again to generate the saved model.")
            return

        model = joblib.load(model_path)

        # Predict
        print("Generating predictions...")
        y_pred = model.predict(X_test)
        
        # Accuracy Check
        acc = accuracy_score(y_test, y_pred)
        print(f"Loaded Model Accuracy: {acc:.4f}")

        # Save CM (Normalized)
        cm_path = os.path.join(OUTPUT_DIR, f'confusion_matrix_{best_backbone}_{best_compression}.png')
        
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(np.concatenate([y_train, y_test]))
        
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        disp.plot(cmap='Blues', values_format='.2f', xticks_rotation='vertical')
        plt.title(f'Normalized Confusion Matrix\n({best_backbone.upper()} + {best_compression.upper()}) Acc: {acc:.2f}')
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Confusion Matrix saved to {cm_path}")
        
        # Save Classification Report
        report = classification_report(y_test, y_pred, target_names=[str(l) for l in unique_labels])
        with open(os.path.join(OUTPUT_DIR, 'classification_report_yolo.txt'), 'w') as f:
            f.write(report)
            
    except Exception as e:
        print(f"Could not generate CM: {e}")
        import traceback
        traceback.print_exc()

# MAIN
if __name__ == "__main__":
    if df is not None:
        plot_accuracy_comparison(df)
    plot_confusion_matrix()
    print("Visualization generation complete.")
