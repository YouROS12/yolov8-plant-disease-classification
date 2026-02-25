import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

print("Generating updated Confusion Matrix for Champion Model (11 Classes)...")

BASE_PATH = '/content/drive/MyDrive/reviewers_suggestion_paper1/'
OUTPUT_DIR = os.path.join(BASE_PATH, 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Pre-Extracted Test Features and Labels
feature_test_file = os.path.join(BASE_PATH, 'x_test_yolo_ipca.npy')
label_test_file = os.path.join(BASE_PATH, 'y_test_11_yolo_ipca.npy')
x_te = np.load(feature_test_file)
y_test = np.load(label_test_file)

# 2. Load the Saved SVC Model
model_filename = 'svm_model_yolo_ipca_11 Classes.joblib'
model_path = os.path.join(BASE_PATH, model_filename)
model = joblib.load(model_path)

# 3. Predict metrics
y_pred = model.predict(x_te)
acc = accuracy_score(y_test, y_pred)
print(f"Verified Champion Accuracy: {acc*100:.2f}%")

# 4. Generate the Confusion Matrix
class_names = [
    'abiotic_disorder', 'bacterial_disease', 'fungal_downy_mildew', 
    'fungal_leaf_disease', 'fungal_powdery_mildew', 'fungal_rot_fruit_disease', 
    'fungal_rust', 'fungal_scab', 'fungal_systemic_smut_gall', 
    'oomycete_lesion', 'viral_disease'
]

cm = confusion_matrix(y_test, y_pred, normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", 
            xticklabels=class_names, yticklabels=class_names,
            cbar=False, annot_kws={"size": 10})

plt.title(f"Champion Model Confusion Matrix (YOLOv8m + IPCA + SVC)\nAccuracy: {acc*100:.2f}%", fontsize=14, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, "champion_confusion_matrix_updated.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Confusion Matrix saved to: {cm_path}")
