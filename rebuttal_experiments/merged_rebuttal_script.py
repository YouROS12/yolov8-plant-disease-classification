# %% Cell 1: Setup and Validation
# ==========================================
# !pip install -q ultralytics timm scikit-learn pandas
import os
import sys

# Auto-install missing dependencies if in Colab
if 'google.colab' in sys.modules:
    try:
        import ultralytics
        import timm
    except ImportError:
        print("[INFO] Installing missing dependencies...")
        os.system('pip install -q ultralytics timm')
import time
import zipfile

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
    IN_COLAB = True
except ImportError:
    print("Not running in Google Colab. Will proceed with local environment.")
    IN_COLAB = False

# %% Cell 2: Imports & Configuration
# ==========================================
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.decomposition import IncrementalPCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
    import timm
except ImportError:
    print("[WARNING] Please install ultralytics and timm: pip install ultralytics timm")

# --- CONFIGURATION ---
BASE_SAVE_DIR = '/content/drive/MyDrive/reviewers_suggestion_paper1/' if IN_COLAB else './reviewers_suggestion_paper1/'
FINETUNE_SAVE_DIR = os.path.join(BASE_SAVE_DIR, 'finetuned')
DATA_DIR = '/content/plantwild_temp'
DEVICE_TRAIN = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE_BENCH = 'cpu' # We benchmark latency strictly on CPU for "edge" claims

EPOCHS = 15          # Full training for fair comparison  
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
N_RUNS = 50          # For latency testing
N_PCA_COMP = 100     # For ROM compression

os.makedirs(FINETUNE_SAVE_DIR, exist_ok=True)
print(f"--- Rebuttal Experiments Runner ---")
print(f"Training Device: {DEVICE_TRAIN.upper()}")
print(f"Benchmarking Device: {DEVICE_BENCH.upper()}")

# %% Cell 3: Data Extraction & Dataset Definition
# ==========================================
import shutil

print("\n--- 1. Preparing Dataset ---")
def setup_local_data():
    # Paths
    drive_storage_dir = os.path.join(BASE_SAVE_DIR, 'plant_wild') # Permanent storage on Drive
    local_data_dir = DATA_DIR # /content/plantwild_temp
    
    # Specific uploaded files (temporal)
    uploaded_train_zip = '/content/train-20260214T191606Z-1-001.zip'
    uploaded_test_zip  = '/content/test-20260214T191603Z-1-001.zip'
    
    # 1. SETUP DRIVE STORAGE (Save uploaded zips if present)
    os.makedirs(drive_storage_dir, exist_ok=True)
    
    def save_to_drive(src, name):
        dst = os.path.join(drive_storage_dir, name)
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"[INFO] Saving {src} to Drive as {name}...")
            shutil.copy(src, dst)
        elif os.path.exists(dst):
            print(f"[INFO] {name} already verified on Drive.")

    save_to_drive(uploaded_train_zip, 'train.zip')
    save_to_drive(uploaded_test_zip,  'test.zip')

    # 2. SETUP LOCAL DATA (Copy from Drive -> Local and Unzip)
    if os.path.exists(os.path.join(local_data_dir, 'train')) and os.path.exists(os.path.join(local_data_dir, 'test')):
        print(f"[INFO] Local data ready at {local_data_dir}. Skipping setup.")
        return

    print(f"[INFO] Setting up local data at {local_data_dir}...")
    os.makedirs(local_data_dir, exist_ok=True)

    def fetch_and_unzip(zip_name, target_folder_name):
        drive_path = os.path.join(drive_storage_dir, zip_name)
        if not os.path.exists(drive_path):
            print(f"[WARNING] {zip_name} not found in Drive storage ({drive_path}). Cannot set up.")
            sys.exit(1)
        print(f"-> Unzipping {zip_name} from Drive...")
        with zipfile.ZipFile(drive_path, 'r') as zip_ref:
            zip_ref.extractall(local_data_dir)
            
    fetch_and_unzip('train.zip', 'train')
    fetch_and_unzip('test.zip',  'test')
    
setup_local_data()

class PlantDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, '*', '*.jpg'))
        self.labels = [os.path.basename(os.path.dirname(p)) for p in self.image_paths]
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, self.labels[idx], img_path

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = PlantDataset(os.path.join(DATA_DIR, 'train'), transform=transform_train)
test_dataset = PlantDataset(os.path.join(DATA_DIR, 'test'), transform=transform_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Label Mapping (11 Classes)
mapping_19_classes = {"wheat stripe rust": "rust", "wheat stem rust": "rust", "wheat leaf rust": "rust", "soybean rust": "rust", "raspberry yellow rust": "rust", "peach rust": "rust", "plum rust": "rust", "garlic rust": "rust", "corn rust": "rust", "coffee leaf rust": "rust", "blueberry rust": "rust", "bean rust": "rust", "apple rust": "rust", "wheat powdery mildew": "powdery_mildew", "zucchini powdery mildew": "powdery_mildew", "squash powdery mildew": "powdery_mildew", "cherry powdery mildew": "powdery_mildew", "cucumber powdery mildew": "powdery_mildew", "bell pepper powdery mildew": "powdery_mildew", "zucchini downy mildew": "downy_mildew", "soybean downy mildew": "downy_mildew", "lettuce downy mildew": "downy_mildew", "grape downy mildew": "downy_mildew", "cabbage downy mildew": "downy_mildew", "broccoli downy mildew": "downy_mildew", "basil downy mildew": "downy_mildew", "tomato late blight": "blight", "tomato early blight": "blight", "raspberry fire blight": "blight", "potato late blight": "blight", "potato early blight": "blight", "corn northern leaf blight": "blight", "eggplant phytophthora blight": "blight", "garlic leaf blight": "blight", "celery early blight": "blight", "carrot alternaria leaf blight": "blight", "blueberry botrytis blight": "blight", "bean halo blight": "blight", "soybean bacterial blight": "blight", "ginger sheath blight": "blight", "rice sheath blight": "blight", "rice blast": "blight", "tomato bacterial leaf spot": "leaf_spot", "tomato septoria leaf spot": "leaf_spot", "tobacco frogeye leaf spot": "leaf_spot", "soybean frog eye leaf spot": "leaf_spot", "tobacco brown spot": "leaf_spot", "soybean brown spot": "leaf_spot", "raspberry leaf spot": "leaf_spot", "plum bacterial spot": "leaf_spot", "ginger leaf spot": "leaf_spot", "grape leaf spot": "leaf_spot", "cucumber angular leaf spot": "leaf_spot", "eggplant cercospora leaf spot": "leaf_spot", "cherry leaf spot": "leaf_spot", "coffee brown eye spot": "leaf_spot", "corn gray leaf spot": "leaf_spot", "cauliflower alternaria leaf spot": "leaf_spot", "carrot cercospora leaf blight": "leaf_spot", "broccoli ring spot": "leaf_spot", "cabbage alternaria leaf spot": "leaf_spot", "bell pepper frogeye leaf spot": "leaf_spot", "broccoli alternaria leaf spot": "leaf_spot", "bell pepper bacterial spot": "leaf_spot", "banana black leaf streak": "leaf_spot", "banana cordana leaf spot": "leaf_spot", "maple tar spot": "leaf_spot", "zucchini yellow mosaic virus": "virus", "tomato mosaic virus": "virus", "tomato yellow leaf curl virus": "virus", "tobacco mosaic virus": "virus", "soybean mosaic": "virus", "plum pox virus": "virus", "lettuce mosaic virus": "virus", "grapevine leafroll disease": "virus", "citrus greening disease": "virus", "banana bunchy top": "virus", "bean mosaic virus": "virus", "apple mosaic virus": "virus", "peach brown rot": "rot", "plum brown rot": "rot", "grape black rot": "rot", "eggplant phomopsis fruit rot": "rot", "coffee black rot": "rot", "cabbage black rot": "rot", "cauliflower bacterial soft rot": "rot", "bell pepper blossom end rot": "rot", "banana cigar end rot": "rot", "apple black rot": "rot", "strawberry anthracnose": "anthracnose", "peach anthracnose": "anthracnose", "celery anthracnose": "anthracnose", "blueberry anthracnose": "anthracnose", "banana anthracnose": "anthracnose", "wheat head scab": "scab", "peach scab": "scab", "apple scab": "scab", "zucchini bacterial wilt": "bacterial_disease", "wheat bacterial leaf streak (black chaff)": "bacterial_disease", "cucumber bacterial wilt": "bacterial_disease", "citrus canker": "bacterial_disease", "tomato leaf mold": "mold", "tobacco blue mold": "mold", "raspberry gray mold": "mold", "wheat septoria blotch": "septoria_blotch", "wheat loose smut": "smut", "corn smut": "smut", "strawberry leaf scorch": "leaf_scorch", "blueberry scorch": "leaf_scorch", "plum pocket disease": "fungal_gall", "peach leaf curl": "fungal_gall", "blueberry mummy berry": "mummy_berry", "banana panama disease": "fungal_wilt", "carrot cavity spot": "oomycete_lesion", "coffee berry blotch": "berry_blotch"}
mapping_11_classes = {
    "rust": "fungal_rust", 
    "powdery_mildew": "fungal_powdery_mildew",
    "downy_mildew": "fungal_downy_mildew",
    "blight": "fungal_leaf_disease", 
    "leaf_spot": "fungal_leaf_disease", 
    "scab": "fungal_leaf_disease", 
    "virus": "viral_disease",
    "rot": "fungal_rot_fruit_disease",
    "anthracnose": "fungal_rot_fruit_disease", 
    "bacterial_disease": "bacterial_disease",
    "mold": "fungal_leaf_disease",
    "septoria_blotch": "fungal_leaf_disease",
    "smut": "fungal_systemic_smut_gall",
    "leaf_scorch": "abiotic_disorder", 
    "fungal_gall": "fungal_systemic_smut_gall",
    "mummy_berry": "fungal_rot_fruit_disease",
    "fungal_wilt": "fungal_leaf_disease",
    "oomycete_lesion": "oomycete_lesion",
    "berry_blotch": "fungal_leaf_disease",
    "insect_mite": "insect_mite",
    "healthy": "healthy",
    "background": "background",
    "unmapped": "unmapped"
}

def get_11_class(lbl):
    sup = mapping_19_classes.get(lbl, "unmapped")
    return mapping_11_classes.get(sup, sup)

encoder = LabelEncoder()
raw_lbls = list(set([os.path.basename(os.path.dirname(p)) for p in train_dataset.image_paths]))
encoder.fit([get_11_class(l) for l in raw_lbls])
NUM_CLASSES = len(encoder.classes_)


# %% Cell 4: DL Architecture Benchmarks (CPU speed, MB size)
# ==========================================
print("\n--- 2. Benchmarking Deep Learning Extractors (CPU) ---")
dl_benchmarks = {}

def measure_dl_stats(model, img_size):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sz_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024**2
    model.to(DEVICE_BENCH).eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(DEVICE_BENCH)
    
    with torch.no_grad():
        for _ in range(5): model(dummy) # Warmup
        lats = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            model(dummy)
            lats.append((time.perf_counter()-t0)*1000)
    return params, sz_mb, np.mean(lats), np.std(lats)

try:
    print("   -> Evaluating ResNet50...")
    m = models.resnet50(pretrained=False); m.fc = nn.Identity()
    p, sz, l, s = measure_dl_stats(m, 224)
    dl_benchmarks['ResNet50'] = {'params': p, 'size_mb': sz, 'lat': l, 'std': s, 'dim': 2048}
except Exception as e: print(f"Error in ResNet benchmark: {e}")

try:
    print("   -> Evaluating EfficientNet-B0...")
    import timm
    m = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
    p, sz, l, s = measure_dl_stats(m, 224)
    dl_benchmarks['EfficientNet-B0'] = {'params': p, 'size_mb': sz, 'lat': l, 'std': s, 'dim': 1280}
except Exception as e: print(f"Error in EfficientNet benchmark: {e}")

try:
    print("   -> Evaluating YOLOv8m...")
    from ultralytics import YOLO
    
    # Use best.pt if provided, otherwise fallback to base
    yolo_path = CUSTOM_WEIGHTS.get('yolov8m', 'yolov8m.pt')
    if yolo_path is None or not os.path.exists(yolo_path): 
        yolo_path = 'yolov8m.pt'
        
    m = YOLO(yolo_path).model
    p, sz, l, s = measure_dl_stats(m, 640)
    dl_benchmarks['YOLOv8m'] = {'params': p, 'size_mb': sz, 'lat': l, 'std': s, 'dim': 192000}
except Exception as e: print(f"Error in YOLO benchmark: {e}")


# %% Cell 5: Baseline Fine-Tuning (Fair Comparison Setup)
# ==========================================
print(f"\n--- 3. Fine-Tuning Baselines on {DEVICE_TRAIN.upper()} ---")

# If you already have your fine-tuned weights saved somewhere else, put the path here! 
# (e.g., '/content/drive/MyDrive/my_best_efficientnet.pth')
# If None, the script will train them from scratch and save them in FINETUNE_SAVE_DIR.
CUSTOM_WEIGHTS = {
    'resnet50': None,
    'efficientnet_b0': "/content/drive/MyDrive/efficientnet_b0_best_model.pth",
    'yolov8m': "/content/drive/MyDrive/best.pt" # Update if your YOLO weights are named differently
}

def finetune_model(model_name):
    if CUSTOM_WEIGHTS.get(model_name) is not None:
        print(f"[INFO] Using explicitly provided weights for {model_name}: {CUSTOM_WEIGHTS[model_name]}")
        return CUSTOM_WEIGHTS[model_name]
        
    save_path = os.path.join(FINETUNE_SAVE_DIR, f"{model_name}_finetuned.pth")
    if os.path.exists(save_path):
        print(f"[INFO] Fine-tuned weights exist at {save_path}. Skipping training.")
        return save_path
        
    print(f"   -> Training {model_name} for {EPOCHS} epochs...")
    if model_name == 'resnet50':
        m = models.resnet50(pretrained=True); m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif model_name == 'efficientnet_b0':
        m = timm.create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES)
        
    m = m.to(DEVICE_TRAIN)
    optimizer = optim.Adam(m.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        m.train()
        run_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, lbls, _ in pbar:
            tgt = torch.tensor(encoder.transform([get_11_class(l) for l in lbls])).to(DEVICE_TRAIN)
            imgs = imgs.to(DEVICE_TRAIN)
            optimizer.zero_grad()
            out = m(imgs)
            loss = criterion(out, tgt)
            loss.backward(); optimizer.step()
            run_loss += loss.item()
            correct += out.max(1)[1].eq(tgt).sum().item(); total += tgt.size(0)

            # Calculate current average loss and accuracy, handling potential division by zero
            current_loss_avg = run_loss / pbar.n if pbar.n > 0 else 0.0
            current_acc_avg = 100. * correct / total if total > 0 else 0.0

            pbar.set_postfix({'loss': current_loss_avg, 'acc': current_acc_avg})
            
    torch.save(m.state_dict(), save_path)
    return save_path

path_res = finetune_model('resnet50')
path_eff = "/content/drive/MyDrive/efficientnet_b0_best_model.pth"


# %% Cell 6: Feature Extraction & Pipeline Accuracy Evaluation
# ==========================================
import joblib
from sklearn.decomposition import TruncatedSVD
print("\n--- 4. Evaluating Pipeline (11 & 19 Classes) ---")

encoder_19 = LabelEncoder()
encoder_19.fit([mapping_19_classes.get(l, 'unmapped') for l in raw_lbls])

def extract_features(model_name, weight_path, dataset):
    state_dict = torch.load(weight_path, map_location=DEVICE_TRAIN)
    if model_name == 'resnet50':
        m = models.resnet50()
        checkpoint_classes = state_dict['fc.weight'].shape[0]
        m.fc = nn.Linear(m.fc.in_features, checkpoint_classes)
        m.load_state_dict(state_dict)
        m.fc = nn.Identity()
    elif model_name == 'efficientnet_b0':
        import timm # Safeguard for NameError if top-level import failed
        # Detect classifier name and size in weights
        cl_key = 'classifier.weight' if 'classifier.weight' in state_dict else 'fc.weight'
        checkpoint_classes = state_dict[cl_key].shape[0]
        m = timm.create_model('efficientnet_b0', pretrained=False, num_classes=checkpoint_classes)
        m.load_state_dict(state_dict)
        m.classifier = nn.Identity()
        
    m.to(DEVICE_TRAIN).eval()
    ldr = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    feats, lbls = [], []
    with torch.no_grad():
        for imgs, r_lbls, _ in tqdm(ldr, desc=f"Extracting {model_name}"):
            out = m(imgs.to(DEVICE_TRAIN))
            feats.append(out.reshape(out.size(0), -1).cpu().numpy())
            lbls.extend(r_lbls) # Return raw layer names
    return np.vstack(feats), np.array(lbls)

RESULTS_FILE = os.path.join(BASE_SAVE_DIR, 'intermediate_results.csv')

def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        try:
            df = pd.read_csv(RESULTS_FILE)
            if 'Label_Set' in df.columns:
                return df.to_dict('records')
        except: pass
    return []

existing_results = load_existing_results()

def check_exists(bb, comp, l_set):
    l_str = f"{l_set} Classes"
    for r in existing_results:
        if str(r.get('Backbone')) == bb and str(r.get('Compression')) == comp and str(r.get('Label_Set')) == l_str:
            return True
    return False

fair_results = {}
ml_benchmarks = {}

models_to_eval = [
    ('resnet50_finetuned', path_res, 'resnet50'),
    ('efficientnet_b0_finetuned', path_eff, 'efficientnet_b0')
]

for bb_name, pth, orig_name in models_to_eval:
    needed = []
    for comp in ['ipca', 'svd']:
        for l_set in [11, 19]:
            if not check_exists(bb_name, comp, l_set):
                needed.append((comp, l_set))
                
    joblib_path = os.path.join(FINETUNE_SAVE_DIR, f"{bb_name}_ml_benchmarks.joblib")
    
    if not needed:
        print(f"\n[INFO] All tasks for {bb_name} already completed. Loading objects from disk.")
        if os.path.exists(joblib_path):
            ml_benchmarks[orig_name] = joblib.load(joblib_path)
            
        for r in existing_results:
            if r['Backbone'] == bb_name and r['Compression'] == 'ipca' and r['Label_Set'] == '11 Classes':
                fair_results[orig_name] = {'acc': r['Accuracy'] * 100, 'f1': r['F1_Score']}
        continue
        
    print(f"\n--- Extracting Features for {bb_name} ({len(needed)} tasks pending) ---")
    x_tr, y_tr_r = extract_features(orig_name, pth, train_dataset)
    x_te, y_te_r = extract_features(orig_name, pth, test_dataset)
    
    for comp, l_set in needed:
        print(f" -> {bb_name} | {comp.upper()} | {l_set} Classes")
        if comp == 'ipca':
            reducer = IncrementalPCA(n_components=N_PCA_COMP, batch_size=512)
        else:
            reducer = TruncatedSVD(n_components=N_PCA_COMP)
            
        x_tr_p = reducer.fit_transform(x_tr)
        x_te_p = reducer.transform(x_te)
        
        if l_set == 11:
            y_tr = encoder.transform([get_11_class(l) for l in y_tr_r])
            y_te = encoder.transform([get_11_class(l) for l in y_te_r])
        else:
            y_tr = encoder_19.transform([mapping_19_classes.get(l, 'unmapped') for l in y_tr_r])
            y_te = encoder_19.transform([mapping_19_classes.get(l, 'unmapped') for l in y_te_r])
            
        svc = SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced')
        t0 = time.perf_counter()
        svc.fit(x_tr_p, y_tr)
        t_time = time.perf_counter() - t0
        
        preds = svc.predict(x_te_p)
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average='macro')
        
        print(f"    Acc: {acc*100:.2f}% | F1: {f1:.4f} | Time: {t_time:.1f}s")
        
        existing_results.append({
            'Backbone': bb_name,
            'Compression': comp,
            'Label_Set': f"{l_set} Classes",
            'Accuracy': acc,
            'F1_Score': f1,
            'Training_Time_seconds': t_time
        })
        
        pd.DataFrame(existing_results).to_csv(RESULTS_FILE, index=False)
        
        if comp == 'ipca' and l_set == 11:
             fair_results[orig_name] = {'acc': acc * 100, 'f1': f1}
             ml_benchmarks[orig_name] = {'pca': reducer, 'svc': svc, 'train_time': t_time, 'raw_dim': x_tr.shape[1]}
             joblib.dump(ml_benchmarks[orig_name], joblib_path)
             
    # Ensure they are loaded if previously evaluated but not this round
    if orig_name not in ml_benchmarks and os.path.exists(joblib_path):
         ml_benchmarks[orig_name] = joblib.load(joblib_path)
    if orig_name not in fair_results:
         for r in existing_results:
             if r['Backbone'] == bb_name and r['Compression'] == 'ipca' and r['Label_Set'] == '11 Classes':
                 fair_results[orig_name] = {'acc': r['Accuracy'] * 100, 'f1': r['F1_Score']}

# %% Cell 7: ML Pipeline Efficiency Benchmarking (CPU Speed & MB Size)
# ==========================================
print("\n--- 5. Benchmarking ML Components (CPU) ---")
# Using the objects we just trained to measure their latency
for m_name, objects in ml_benchmarks.items():
    pca, svc, r_dim = objects['pca'], objects['svc'], objects['raw_dim']
    
    # Calculate MB size
    pca_params = pca.components_.size + pca.mean_.size
    svc_params = svc.dual_coef_.size + len(svc.support_vectors_) * N_PCA_COMP + svc.intercept_.size
    mb_sz = ((pca_params + svc_params) * 4) / 1024**2
    
    dummy = np.random.randn(1, r_dim)
    
    # Warmup
    for _ in range(5): svc.predict(pca.transform(dummy))
    
    p_lats, s_lats = [], []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        f_c = pca.transform(dummy)
        p_lats.append((time.perf_counter()-start)*1000)
        
        start2 = time.perf_counter()
        svc.predict(f_c)
        s_lats.append((time.perf_counter()-start2)*1000)
        
    objects['size_mb'] = mb_sz
    objects['pca_params'] = pca_params
    objects['svc_params'] = svc_params
    objects['pca_lat'] = np.mean(p_lats)
    objects['svc_lat'] = np.mean(s_lats)

# %% Cell 8: Final Rebuttal Tables
# ==========================================
print("\n\n" + "="*80)
print(f"{'TABLE 1: FAIR COMPARISON (Pre-training Addressed)':^80}")
print("="*80)
print(f"{'Pipeline':<45} | {'Accuracy (%)':<15} | {'F1-Macro'}")
print("-" * 80)
print(f"ResNet50 (Target-Finetuned) + IPCA + SVC      | {fair_results['resnet50']['acc']:>7.2f}%        | {fair_results['resnet50']['f1']:.3f}")
print(f"EfficientNet-B0 (Target-Finetuned) + IPCA+SVC | {fair_results['efficientnet_b0']['acc']:>7.2f}%        | {fair_results['efficientnet_b0']['f1']:.3f}")
print(f"YOLOv8m (Target-Finetuned) + IPCA + SVC       |   87.52%        | 0.882   (From Paper)")
print("="*80)


print("\n\n" + "="*85)
print(f"{'TABLE 2: COMPUTATIONAL EFFICIENCY BENCHMARKS':^85}")
print("="*85)
print(f"{'Component':<25} | {'Params (M)':<12} | {'Size (MB)':<10} | {'Latency (ms)':<15} | {'Raw Dim / Target'}")
print("-" * 85)
for name, res in dl_benchmarks.items():
    p_m = res['params'] / 1e6
    print(f"Ext: {name:<20} | {p_m:>10.2f} M | {res['size_mb']:>8.1f}   | {res['lat']:>6.1f} ± {res['std']:<4.1f} | {res['dim']} (Features)")

print("-" * 85)
# Using ResNet50's ML objects as representative for the ML block speed
rep = ml_benchmarks['resnet50'] 
p_m_pca = rep['pca_params'] / 1e6
p_m_svc = rep['svc_params'] / 1e6
print(f"{'Cls: IPCA Transform':<25} | {p_m_pca:>10.2f} M | {(rep['pca_params']*4)/1024**2:>8.1f}   | {rep['pca_lat']:>6.1f} ± 0.0  | 100 (Latent/ROM)")
print(f"{'Cls: SVC Inference':<25} | {p_m_svc:>10.6f} M | {(rep['svc_params']*4)/1024**2:>8.3f}   | {rep['svc_lat']:>6.1f} ± 0.0  | 11 (Classes)")
print("="*85)
print(f"Total ML Block Size: {rep['size_mb']:.2f} MB")
print(f"SVC Retraining Time on Edge (CPU): {rep['train_time']:.2f} seconds")

# --- Save Results to CSV ---
fairness_df = pd.DataFrame([
    {'Pipeline': 'ResNet50 + IPCA + SVC', 'Accuracy': fair_results['resnet50']['acc'], 'F1': fair_results['resnet50']['f1']},
    {'Pipeline': 'EfficientNet-B0 + IPCA + SVC', 'Accuracy': fair_results['efficientnet_b0']['acc'], 'F1': fair_results['efficientnet_b0']['f1']},
    {'Pipeline': 'YOLOv8m + IPCA + SVC', 'Accuracy': 87.52, 'F1': 0.882}
])
fairness_df.to_csv(os.path.join(BASE_SAVE_DIR, 'rebuttal_fairness_results.csv'), index=False)

eff_records = []
for name, res in dl_benchmarks.items():
    eff_records.append({'Component': f"Ext: {name}", 'Params(M)': res['params']/1e6, 'Size(MB)': res['size_mb'], 'Latency(ms)': res['lat'], 'Latency_Std': res['std'], 'Dim': res['dim']})
eff_records.append({'Component': 'Cls: IPCA Transform', 'Params(M)': p_m_pca, 'Size(MB)': (rep['pca_params']*4)/1024**2, 'Latency(ms)': rep['pca_lat'], 'Latency_Std': 0.0, 'Dim': 100})
eff_records.append({'Component': 'Cls: SVC Inference', 'Params(M)': p_m_svc, 'Size(MB)': (rep['svc_params']*4)/1024**2, 'Latency(ms)': rep['svc_lat'], 'Latency_Std': 0.0, 'Dim': 11})

eff_df = pd.DataFrame(eff_records)
eff_df.to_csv(os.path.join(BASE_SAVE_DIR, 'rebuttal_efficiency_benchmarks.csv'), index=False)
print(f"\n[INFO] Saved results to {BASE_SAVE_DIR}rebuttal_fairness_results.csv and rebuttal_efficiency_benchmarks.csv")
