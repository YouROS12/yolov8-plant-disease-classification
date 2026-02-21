# %% Cell 1: Setup and Validation
# ==========================================
# Install necessary packages
# !pip install ultralytics timm

from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Verify the mount
if os.path.exists('/content/drive/MyDrive'):
    print("Google Drive mounted successfully.")
else:
    print("Error: Google Drive not mounted.")

# %% Cell 2: Imports & Configuration
# ==========================================
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from collections import Counter
import warnings
from ultralytics import YOLO
import joblib
import timm
from sklearn.decomposition import IncrementalPCA, TruncatedSVD

warnings.filterwarnings("ignore")

# --- EXPERIMENT CONTROL PANEL ---
# ==========================================
# Uncomment ANY experiments you want to run.
# They will run sequentially. If one fails, the next will still run.
# ==========================================

EXPERIMENTS = [
    # (Backbone, Compression)
    ('resnet50',        'ipca'),   # Run 1
    ('resnet50',        'svd'),    # Run 2
    ('efficientnet_b0', 'ipca'),   # Run 3
    ('efficientnet_b0', 'svd'),    # Run 4
    ('yolo',            'ipca'),   # Run 5
    ('yolo',            'svd')    # Run 6
]

# --- SHARED CONFIGURATION ---
# --- SHARED CONFIGURATION ---
BASE_CONFIG = {
    'N_COMPONENTS': 100,
    'N_COMPONENTS': 100,
    # Tip: Point this to a .zip file on Drive for faster copying! (e.g. '/content/drive/MyDrive/data.zip')
    # Copying a folder with thousands of images takes a long time on Drive.
    'Generic_Base': '/content/drive/MyDrive/plantwild_split', 
    'USE_LOCAL_COPY': True,  # Set to False to skip copy and run slow (directly from Drive)
    'BASE_DIR': '/content/plantwild_temp', # Will act as new local base
    'SAVE_DIR': '/content/drive/MyDrive/reviewers_suggestion_paper1/',
    'YOLO_WEIGHTS': '/content/drive/MyDrive/plantwild_split/best.pt',
    'YOLO_FEATURE_LAYER': 8,  # Matches published results
}

# Ensure save directory exists
os.makedirs(BASE_CONFIG['SAVE_DIR'], exist_ok=True)

# Device Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- DATA TRANSFER SETUP ---
import shutil
import zipfile

def setup_local_data():
    # Paths
    drive_storage_dir = os.path.join(BASE_CONFIG['SAVE_DIR'], 'plant_wild') # Permanent storage on Drive
    local_data_dir = BASE_CONFIG['BASE_DIR'] # /content/plantwild_temp
    
    # Specific uploaded files (temporal)
    uploaded_train_zip = '/content/train-20260214T191606Z-1-001.zip'
    uploaded_test_zip  = '/content/test-20260214T191603Z-1-001.zip'
    
    # 1. SETUP DRIVE STORAGE (Save uploaded zips if present)
    os.makedirs(drive_storage_dir, exist_ok=True)
    
    # Function to save a zip to Drive renaming it for consistency
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

    # Function to fetch and unzip
    def fetch_and_unzip(zip_name, target_folder_name):
        drive_path = os.path.join(drive_storage_dir, zip_name)
        
        if not os.path.exists(drive_path):
            print(f"[WARNING] {zip_name} not found in Drive storage ({drive_path}). Cannot set up.")
            return

        print(f"-> Unzipping {zip_name} from Drive...")
        with zipfile.ZipFile(drive_path, 'r') as zip_ref:
            zip_ref.extractall(local_data_dir)
            # Check if it extracted to a nested folder or directly
            # We expect 'train' and 'test' folders to appear in local_data_dir
    
    fetch_and_unzip('train.zip', 'train')
    fetch_and_unzip('test.zip',  'test')
    
    print(f"[INFO] Setup complete. Contents of {local_data_dir}: {os.listdir(local_data_dir)}")

setup_local_data()

# %% Cell 3: Feature Extraction Helpers (Backbone Zoo)
# ==========================================
intermediate_features = []

def hook_fn(module, input, output):
    intermediate_features.append(output)

def _select_yolo_feature_layer(yolo_model, prefer_idx=None):
    modules = yolo_model.model.model
    if prefer_idx is not None:
        idx = prefer_idx if prefer_idx >= 0 else len(modules) + prefer_idx
        if 0 <= idx < len(modules):
            return modules[idx], idx

    # Try to select the last non-detect, backbone-like block
    for i in range(len(modules) - 1, -1, -1):
        name = modules[i].__class__.__name__.lower()
        if 'detect' in name:
            continue
        if any(k in name for k in ['c2f', 'c3', 'sppf', 'conv', 'bottleneck']):
            return modules[i], i

    # Fallback: choose the second-to-last module if possible
    if len(modules) >= 2:
        return modules[-2], len(modules) - 2
    return modules[-1], len(modules) - 1

def get_feature_extractor(backbone_name):
    """Returns the model and the extraction function for the chosen backbone."""
    
    if backbone_name == 'yolo':
        if not os.path.exists(BASE_CONFIG['YOLO_WEIGHTS']):
            raise FileNotFoundError(f"YOLO weights not found at {BASE_CONFIG['YOLO_WEIGHTS']}")
        model = YOLO(BASE_CONFIG['YOLO_WEIGHTS'])
        # YOLO doesn't need .eval() on the wrapper, but we handle it internally
        
        if BASE_CONFIG.get('YOLO_FEATURE_LAYER') is not None:
            print(f"[INFO] FORCING YOLO feature extraction at user-specified layer: {BASE_CONFIG['YOLO_FEATURE_LAYER']}")

        yolo_layer, yolo_layer_idx = _select_yolo_feature_layer(
            model,
            BASE_CONFIG.get('YOLO_FEATURE_LAYER')
        )
        print(f"YOLO feature layer selected: index={yolo_layer_idx}, type={yolo_layer.__class__.__name__}")

        def extract_yolo(img_tensor):
            global intermediate_features
            intermediate_features = []
            hook = yolo_layer.register_forward_hook(hook_fn) 
            with torch.no_grad():
                model(img_tensor)
            hook.remove()
            return intermediate_features[0]
            
        return model, extract_yolo
        
    elif backbone_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Identity() # Remove classification layer
        model.to(device)
        model.eval()
        
        def extract_resnet(img_tensor):
            with torch.no_grad():
                return model(img_tensor)
        return model, extract_resnet

    elif backbone_name == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0) # num_classes=0 removes classifier
        model.to(device)
        model.eval()
        
        def extract_effnet(img_tensor):
            with torch.no_grad():
                return model(img_tensor)
        return model, extract_effnet
        
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

def preprocess_image(img_path, backbone_name):
    """Preprocesses image according to backbone requirements."""
    size = (640, 640) if backbone_name == 'yolo' else (224, 224)
    
    if backbone_name == 'yolo':
        # YOLO models typically expect 0-1 scaled inputs
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
    
    try:
        img = Image.open(img_path).convert('RGB')
        return transform(img)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

def process_batch(paths, labels, backbone, extractor, batch_size=32):
    features_list = []
    labels_list = []
    
    for i in tqdm(range(0, len(paths), batch_size), desc="Extracting Features"):
        batch_paths = paths[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # Preprocess
        imgs = [preprocess_image(p, backbone) for p in batch_paths]
        
        # Filter None
        valid_imgs = []
        valid_labels = []
        for img, lbl in zip(imgs, batch_labels):
            if img is not None:
                valid_imgs.append(img)
                valid_labels.append(lbl)
                
        if not valid_imgs: continue
        
        # Inference
        batch_tensor = torch.stack(valid_imgs).to(device)
        feats = extractor(batch_tensor)
        if i == 0:
            print(f"[DEBUG] First batch feature tensor shape: {tuple(feats.shape)}")
        
        # Flatten and store
        feats_flat = feats.reshape(feats.size(0), -1).cpu().numpy()
        features_list.extend(list(feats_flat))
        labels_list.extend(valid_labels)
        
    return features_list, labels_list

def process_yolo_batches(paths, labels, model, yolo_layer):
    """
    Robust generator for YOLO feature extraction.
    Yields (features_tensor, valid_labels_list) for each batch.
    Handles potential mismatches by truncating.
    """
    batch_size = 32
    for i in tqdm(range(0, len(paths), batch_size), desc="YOLO Extraction"):
        batch_paths = paths[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # Reuse existing preprocess_image, forcing 'yolo' mode
        imgs = [preprocess_image(p, 'yolo') for p in batch_paths]
        
        valid_imgs = []
        valid_labels = []
        for img, lbl in zip(imgs, batch_labels):
            if img is not None:
                valid_imgs.append(img)
                valid_labels.append(lbl)
                
        if not valid_imgs: continue
        
        global intermediate_features
        intermediate_features = []
        
        batch_tensor = torch.stack(valid_imgs).to(device)
        
        hook = yolo_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
             # YOLO model call
             model(batch_tensor)
        hook.remove()
        
        if not intermediate_features:
            print(f"[WARNING] No features captured for batch starting at {i}.")
            continue
            
        # Concatenate intermediate features if split
        try:
            if len(intermediate_features) > 1:
                feats = torch.cat(intermediate_features, dim=0)
            else:
                feats = intermediate_features[0]
        except Exception as e:
            print(f"[ERROR] Failed to concat features: {e}")
            continue
            
        # Robust Length Check
        if feats.shape[0] != len(valid_labels):
            print(f"[MISMATCH] Batch {i}: Feats={feats.shape[0]}, Labels={len(valid_labels)}. Truncating.")
            min_len = min(feats.shape[0], len(valid_labels))
            feats = feats[:min_len]
            valid_labels = valid_labels[:min_len]
            
        # Yield as CPU numpy (flattened) or tensor? process_batch returns list of floats.
        # But for IPCA/SVD we need 2D array.
        # Let's yield the 2D CPU tensor/numpy to be efficient.
        feats_flat = feats.reshape(feats.size(0), -1).cpu().numpy()
        yield feats_flat, valid_labels

# %% Cell 4: Visual Class Mappings (Experiment 3 Setup)
# ==========================================
# 19 Classes (Visual Similarity)
mapping_19_classes = {
    "wheat stripe rust": "rust", "wheat stem rust": "rust", "wheat leaf rust": "rust", "soybean rust": "rust", "raspberry yellow rust": "rust", "peach rust": "rust", "plum rust": "rust", "garlic rust": "rust", "corn rust": "rust", "coffee leaf rust": "rust", "blueberry rust": "rust", "bean rust": "rust", "apple rust": "rust",
    "wheat powdery mildew": "powdery_mildew", "zucchini powdery mildew": "powdery_mildew", "squash powdery mildew": "powdery_mildew", "cherry powdery mildew": "powdery_mildew", "cucumber powdery mildew": "powdery_mildew", "bell pepper powdery mildew": "powdery_mildew",
    "zucchini downy mildew": "downy_mildew", "soybean downy mildew": "downy_mildew", "lettuce downy mildew": "downy_mildew", "grape downy mildew": "downy_mildew", "cabbage downy mildew": "downy_mildew", "broccoli downy mildew": "downy_mildew", "basil downy mildew": "downy_mildew",
    "tomato late blight": "blight", "tomato early blight": "blight", "raspberry fire blight": "blight", "potato late blight": "blight", "potato early blight": "blight", "corn northern leaf blight": "blight", "eggplant phytophthora blight": "blight", "garlic leaf blight": "blight", "celery early blight": "blight", "carrot alternaria leaf blight": "blight", "blueberry botrytis blight": "blight", "bean halo blight": "blight", "soybean bacterial blight": "blight", "ginger sheath blight": "blight", "rice sheath blight": "blight", "rice blast": "blight",
    "tomato bacterial leaf spot": "leaf_spot", "tomato septoria leaf spot": "leaf_spot", "tobacco frogeye leaf spot": "leaf_spot", "soybean frog eye leaf spot": "leaf_spot", "tobacco brown spot": "leaf_spot", "soybean brown spot": "leaf_spot", "raspberry leaf spot": "leaf_spot", "plum bacterial spot": "leaf_spot", "ginger leaf spot": "leaf_spot", "grape leaf spot": "leaf_spot", "cucumber angular leaf spot": "leaf_spot", "eggplant cercospora leaf spot": "leaf_spot", "cherry leaf spot": "leaf_spot", "coffee brown eye spot": "leaf_spot", "corn gray leaf spot": "leaf_spot", "cauliflower alternaria leaf spot": "leaf_spot", "carrot cercospora leaf blight": "leaf_spot", "broccoli ring spot": "leaf_spot", "cabbage alternaria leaf spot": "leaf_spot", "bell pepper frogeye leaf spot": "leaf_spot", "broccoli alternaria leaf spot": "leaf_spot", "bell pepper bacterial spot": "leaf_spot", "banana black leaf streak": "leaf_spot", "banana cordana leaf spot": "leaf_spot", "maple tar spot": "leaf_spot",
    "zucchini yellow mosaic virus": "virus", "tomato mosaic virus": "virus", "tomato yellow leaf curl virus": "virus", "tobacco mosaic virus": "virus", "soybean mosaic": "virus", "plum pox virus": "virus", "lettuce mosaic virus": "virus", "grapevine leafroll disease": "virus", "citrus greening disease": "virus", "banana bunchy top": "virus", "bean mosaic virus": "virus", "apple mosaic virus": "virus",
    "peach brown rot": "rot", "plum brown rot": "rot", "grape black rot": "rot", "eggplant phomopsis fruit rot": "rot", "coffee black rot": "rot", "cabbage black rot": "rot", "cauliflower bacterial soft rot": "rot", "bell pepper blossom end rot": "rot", "banana cigar end rot": "rot", "apple black rot": "rot",
    "strawberry anthracnose": "anthracnose", "peach anthracnose": "anthracnose", "celery anthracnose": "anthracnose", "blueberry anthracnose": "anthracnose", "banana anthracnose": "anthracnose",
    "wheat head scab": "scab", "peach scab": "scab", "apple scab": "scab",
    "zucchini bacterial wilt": "bacterial_disease", "wheat bacterial leaf streak (black chaff)": "bacterial_disease", "cucumber bacterial wilt": "bacterial_disease", "citrus canker": "bacterial_disease",
    "tomato leaf mold": "mold", "tobacco blue mold": "mold", "raspberry gray mold": "mold",
    "wheat septoria blotch": "septoria_blotch",
    "wheat loose smut": "smut", "corn smut": "smut",
    "strawberry leaf scorch": "leaf_scorch", "blueberry scorch": "leaf_scorch",
    "plum pocket disease": "fungal_gall", "peach leaf curl": "fungal_gall",
    "blueberry mummy berry": "mummy_berry",
    "banana panama disease": "fungal_wilt",
    "carrot cavity spot": "oomycete_lesion",
    "coffee berry blotch": "berry_blotch",
}

# 11 Classes (Treatment/Type Based)
mapping_11_classes = {
    "rust": "fungal_rust", 
    "powdery_mildew": "fungal_powdery_mildew",
    "downy_mildew": "fungal_downy_mildew",
    "blight": "fungal_leaf_disease", # Includes all blights
    "leaf_spot": "fungal_leaf_disease", # Includes all spots
    "scab": "fungal_leaf_disease", # Scabs are arguably leaf/fruit diseases
    "virus": "viral_disease",
    "rot": "fungal_rot_fruit_disease",
    "anthracnose": "fungal_rot_fruit_disease", # Includes fruit rot
    "bacterial_disease": "bacterial_disease",
    "mold": "fungal_leaf_disease",
    "septoria_blotch": "fungal_leaf_disease",
    "smut": "fungal_systemic_smut_gall",
    "leaf_scorch": "abiotic_disorder", # Scorch is often abiotic/fungal but we group it
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

# Helper to apply 2-stage mapping for 11 classes
def get_11_class_label(original_label):
    super_class = mapping_19_classes.get(original_label, "unmapped")
    return mapping_11_classes.get(super_class, super_class)

# %% Cell 5: Execution Loop
# ==========================================

print(f"Starting Batch Execution for {len(EXPERIMENTS)} experiments...")

for exp_idx, (backbone_name, compression_method) in enumerate(EXPERIMENTS):
    print(f"\n{'='*40}")
    print(f"RUNNING EXPERIMENT {exp_idx+1}/{len(EXPERIMENTS)}")
    print(f"Backbone: {backbone_name} | Compression: {compression_method}")
    print(f"{'='*40}")

    # Check if already done
    suffix = f"_{backbone_name}_{compression_method}"
    expected_output = os.path.join(BASE_CONFIG['SAVE_DIR'], f'x_train{suffix}.npy')
    
    if os.path.exists(expected_output):
         print(f"[INFO] Results for {suffix} already exist at {expected_output}. SKIPPING.")
         continue

    try:
        # 1. Setup Model
        print(f"--- Setting up Backbone: {backbone_name} ---")
        model, extract_fn = get_feature_extractor(backbone_name)

        # 2. Gather Paths
        all_train_paths, all_train_labels = [], []
        all_test_paths, all_test_labels = [], []

        base_dir = BASE_CONFIG['BASE_DIR']
        train_files = glob.glob(os.path.join(base_dir, 'train', '*', '*.jpg'))
        test_files = glob.glob(os.path.join(base_dir, 'test', '*', '*.jpg'))

        for p in train_files:
            label = os.path.basename(os.path.dirname(p))
            all_train_paths.append(p)
            all_train_labels.append(label)

        for p in test_files:
            label = os.path.basename(os.path.dirname(p))
            all_test_paths.append(p)
            all_test_labels.append(label)
            
        print(f"Found {len(all_train_paths)} train, {len(all_test_paths)} test images.")
        
        if len(all_train_paths) == 0 or len(all_test_paths) == 0:
            print("WARNING: No images found. Skipping experiment.")
            continue

        # 3. Extract Features
        print("Extracting features (this might take time)...")
        
        if backbone_name == 'yolo':
            # Robust YOLO Extraction
            print("Using Robust YOLO Generator...")
            # Select Layer again for the main loop scope
            yolo_layer, _ = _select_yolo_feature_layer(model, BASE_CONFIG.get('YOLO_FEATURE_LAYER'))
            
            # Train
            x_train_batches = []
            train_lbls_raw = []
            for feats, lbls in process_yolo_batches(all_train_paths, all_train_labels, model, yolo_layer):
                x_train_batches.append(feats)
                train_lbls_raw.extend(lbls)
            
            if x_train_batches:
                x_train_np = np.vstack(x_train_batches)
            else:
                x_train_np = np.array([])
                
            # Test
            x_test_batches = []
            test_lbls_raw = []
            for feats, lbls in process_yolo_batches(all_test_paths, all_test_labels, model, yolo_layer):
                x_test_batches.append(feats)
                test_lbls_raw.extend(lbls)
                
            if x_test_batches:
                x_test_np = np.vstack(x_test_batches)
            else:
                x_test_np = np.array([])
                
        else:
            # Standard Extraction (ResNet/EfficientNet)
            x_train_raw, train_lbls_raw = process_batch(all_train_paths, all_train_labels, backbone_name, extract_fn)
            x_test_raw, test_lbls_raw = process_batch(all_test_paths, all_test_labels, backbone_name, extract_fn)

            x_train_np = np.array(x_train_raw)
            x_test_np = np.array(x_test_raw)
        x_test_np = np.array(x_test_raw)
        print(f"Raw Features Shape: {x_train_np.shape}")
        
        if x_train_np.size == 0:
            raise ValueError("No training features extracted.")

        # 4. Compression
        if compression_method == 'ipca':
            print(f"Fitting IncrementalPCA (n={BASE_CONFIG['N_COMPONENTS']})...")
            compressor = IncrementalPCA(n_components=BASE_CONFIG['N_COMPONENTS'], batch_size=512)
            compressor.fit(x_train_np)
            
        elif compression_method == 'svd':
            print(f"Fitting TruncatedSVD (n={BASE_CONFIG['N_COMPONENTS']})...")
            compressor = TruncatedSVD(n_components=BASE_CONFIG['N_COMPONENTS'])
            compressor.fit(x_train_np)

        x_train_compressed = compressor.transform(x_train_np)
        x_test_compressed = compressor.transform(x_test_np)
        print(f"Compressed Shapes: train={x_train_compressed.shape}, test={x_test_compressed.shape}")

        # 5. Label Processing
        from sklearn.preprocessing import LabelEncoder
        
        # 19 Classes
        labels_19_train = [mapping_19_classes.get(l, 'unmapped') for l in train_lbls_raw]
        labels_19_test = [mapping_19_classes.get(l, 'unmapped') for l in test_lbls_raw]
        enc_19 = LabelEncoder()
        y_train_19 = enc_19.fit_transform(labels_19_train)
        y_test_19 = enc_19.transform(labels_19_test)

        # 11 Classes
        labels_11_train = [get_11_class_label(l) for l in train_lbls_raw]
        labels_11_test = [get_11_class_label(l) for l in test_lbls_raw]
        enc_11 = LabelEncoder()
        y_train_11 = enc_11.fit_transform(labels_11_train)
        y_test_11 = enc_11.transform(labels_11_test)

        # 6. Saving
        suffix = f"_{backbone_name}_{compression_method}"
        save_dir = BASE_CONFIG['SAVE_DIR']
        print(f"Saving files with suffix: {suffix} to {save_dir}")

        np.save(os.path.join(save_dir, f'x_train{suffix}.npy'), x_train_compressed)
        np.save(os.path.join(save_dir, f'x_test{suffix}.npy'), x_test_compressed)

        # Labels (Save every time to be safe, though they are identical per dataset)
        np.save(os.path.join(save_dir, f'y_train_19{suffix}.npy'), y_train_19)
        np.save(os.path.join(save_dir, f'y_test_19{suffix}.npy'), y_test_19)
        np.save(os.path.join(save_dir, f'y_train_11{suffix}.npy'), y_train_11)
        np.save(os.path.join(save_dir, f'y_test_11{suffix}.npy'), y_test_11)

        joblib.dump(enc_19, os.path.join(save_dir, f'encoder_19{suffix}.joblib'))
        joblib.dump(enc_11, os.path.join(save_dir, f'encoder_11{suffix}.joblib'))
        joblib.dump(compressor, os.path.join(save_dir, f'compressor{suffix}.joblib'))

        print(f"SUCCESS: Experiment {suffix} completed.")

    except Exception as e:
        print(f"ERROR in experiment {backbone_name} + {compression_method}: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing to next experiment...")

print("\nAll experiments finished.")
