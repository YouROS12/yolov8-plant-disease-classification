# %% Cell 1: Setup & Imports
# ==========================================
import os
import glob
import re
import time
import joblib
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC

import torch
import torch.nn as nn
from torchvision import models
import timm

warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print("Libraries imported.")

# %% Cell 2: Automated Batch Discovery
# ==========================================
# Look for output files from preprocessing.py# CONFIG
BASE_PATH = '/content/drive/MyDrive/reviewers_suggestion_paper1/'
RESULTS_PATH = BASE_PATH  # Save everything in the same folder
RESULTS_FILE = os.path.join(RESULTS_PATH, 'intermediate_results.csv')
SEED = 42

os.makedirs(RESULTS_PATH, exist_ok=True)

print(f"Scanning for experiments in: {BASE_PATH}")

# Discover experiments by looking for x_train_*.npy files
# Expected format: x_train_{backbone}_{compression}.npy
feature_files = glob.glob(os.path.join(BASE_PATH, 'x_train_*.npy'))
experiments = []

for f in feature_files:
    fname = os.path.basename(f)
    # Parse filename: x_train_resnet50_ipca.npy -> backbone=resnet50, compression=ipca
    match = re.search(r'x_train_(.+?)_(ipca|svd)\.npy', fname)
    if match:
        backbone = match.group(1)
        compression = match.group(2)
        experiments.append((backbone, compression))

# Remove duplicates and sort
experiments = sorted(list(set(experiments)))

print(f"\nFound {len(experiments)} unique experiments to process:")
for i, (bb, comp) in enumerate(experiments):
    print(f"  {i+1}. {bb} + {comp}")

if not experiments:
    print("\n[WARNING] No feature files found! Did preprocessing.py finish running?")
    print(f"Checked path: {BASE_PATH}")

# %% Cell 3: Batch Training & Evaluation Loop
# ==========================================
RESULTS_FILE = os.path.join(RESULTS_PATH, 'intermediate_results.csv')
all_results = []

# Load existing results if available (Resume capability)
if os.path.exists(RESULTS_FILE):
    try:
        df_existing = pd.read_csv(RESULTS_FILE)
        # Check for compatibility (Label_Set column)
        if 'Label_Set' not in df_existing.columns:
            print("[WARNING] Existing results are from old version (missing 'Label_Set'). Starting fresh.")
            os.rename(RESULTS_FILE, RESULTS_FILE + '.bak') # Backup
            all_results = []
        else:
            all_results = df_existing.to_dict('records')
            print(f"[INFO] Resuming from {len(all_results)} existing results in {RESULTS_FILE}")
    except Exception as e:
        print(f"[WARNING] Could not load existing results: {e}. Starting fresh.")
        all_results = []

LABEL_SETS_TO_TEST = [19, 11] # 19=Specific Disease, 11=Treatment Grouping (Biological)

for exp_idx, (backbone, compression) in enumerate(experiments):
    print(f"\n{'='*60}")
    print(f"PROCESSING EXPERIMENT {exp_idx+1}/{len(experiments)}")
    print(f"Backbone: {backbone} | Compression: {compression}")
    print(f"{'='*60}")
    
    try:
        needed_labels = []
        for label_count in LABEL_SETS_TO_TEST:
            already_done = any(
                r['Backbone'] == backbone and 
                r['Compression'] == compression and 
                r['Label_Set'] == f"{label_count} Classes" 
                for r in all_results
            )
            if not already_done:
                needed_labels.append(label_count)
                
        if not needed_labels:
            print(f"[INFO] All label sets for {backbone}+{compression} already done. SKIPPING.")
            continue

        # 1. Load Features
        suffix = f"_{backbone}_{compression}"
        feat_file = os.path.join(BASE_PATH, f'x_train{suffix}.npy')
        print(f"Loading features: {feat_file} ...")
        
        if not os.path.exists(feat_file):
            print(f"[ERROR] Feature file not found: {feat_file}. Skipping.")
            continue

        x_train = np.load(feat_file)
        x_test = np.load(os.path.join(BASE_PATH, f'x_test{suffix}.npy'))
        print(f"Features Loaded: Train={x_train.shape}, Test={x_test.shape}")
        
        # 2. Iterate over NEEDED Label Sets
        for label_count in needed_labels:
            print(f"\n--- Evaluating on {label_count}-Class Labels ({'Treatment' if label_count==11 else 'Specific'}) ---")
            
            # Load Labels
            label_train_file_name = f'y_train_{label_count}{suffix}.npy'
            label_train_path = os.path.join(BASE_PATH, label_train_file_name)
            label_test_path = os.path.join(BASE_PATH, f'y_test_{label_count}{suffix}.npy')
            
            if not os.path.exists(label_train_path):
                 print(f"[WARNING] Label file {label_train_file_name} not found. Skipping.")
                 continue

            y_train = np.load(label_train_path)
            y_test = np.load(label_test_path)
            
            # Validation: Shape Mismatch Check
            if len(x_train) != len(y_train):
                print(f"[ERROR] Shape Mismatch!")
                print(f"  Features: {len(x_train)}")
                print(f"  Labels:   {len(y_train)}  (Loaded from: {label_train_file_name})")
                print("  -> Preprocessing likely dropped failing images but labels were not updated.")
                print("  -> Suggestion: Delete this label file and re-run preprocessing.")
                continue

            # 3. Train SVC
            print("Training SVC (Balanced)...")
            start_time = time.time()
            
            model = SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=SEED)
            model.fit(x_train, y_train)
            
            train_time = time.time() - start_time
            print(f"Training finished in {train_time:.2f}s")
            
            # 4. Evaluate
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            
            print(f"-> Accuracy ({label_count}-class): {acc:.4f}")
            print(f"-> F1 Score ({label_count}-class): {f1:.4f}")
            
            # 5. Store Results
            all_results.append({
                'Backbone': backbone,
                'Compression': compression,
                'Label_Set': f"{label_count} Classes",
                'Accuracy': acc,
                'F1_Score': f1,
                'Training_Time_s': train_time
            })
            
            # Save Model
            model_filename = f"svm_model_{backbone}_{compression}_{label_count} Classes.joblib"
            model_path = os.path.join(RESULTS_PATH, model_filename)
            joblib.dump(model, model_path)
            print(f"Model saved to: {model_filename}")
            
            pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
        
    except Exception as e:
        print(f"ERROR processing experiment {backbone}+{compression}: {e}")
        import traceback
        traceback.print_exc()

# %% Cell 4: Final Summary & Export
# ==========================================
if all_results:
    final_df = pd.DataFrame(all_results)
    
    # Sort for better readability
    final_df = final_df.sort_values(by=['Backbone', 'Compression'])
    
    # Save to CSV
    final_csv_path = os.path.join(RESULTS_PATH, 'final_results_summary.csv')
    final_df.to_csv(final_csv_path, index=False)
    
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(final_df.to_markdown(index=False))
    print(f"\nSaved to: {final_csv_path}")
else:
    print("\nNo results generated.")

# %% Cell 5: Speed Benchmark (Optional)
# ==========================================
# Benchmarks CPU inference speed for one backbone (e.g. ResNet50)
print("\n--- Starting Speed Check (Optional Benchmark) ---")

device = 'cpu'
model_name = 'resnet50' # Default
print(f"Benchmarking inference speed for: {model_name}")

try:
    if model_name == 'resnet50':
        model_bench = models.resnet50()
    elif model_name == 'efficientnet_b0':
        model_bench = timm.create_model('efficientnet_b0', pretrained=False)
    else:
        model_bench = None

    if model_bench:
        model_bench.to(device)
        model_bench.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        # Warmup
        for _ in range(5):
            model_bench(dummy_input)

        # Measure
        iterations = 20
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                t0 = time.time()
                model_bench(dummy_input)
                t1 = time.time()
                times.append((t1 - t0) * 1000) # ms

        print(f"Average Inference Time ({model_name}): {np.mean(times):.2f} ms")
except Exception as e:
    print(f"Benchmark failed: {e}")
