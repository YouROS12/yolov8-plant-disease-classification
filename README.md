# YOLOv8-Based Plant Disease Classification

> **Resource-Efficient Framework for Plant Disease Classification using YOLOv8m, PCA, and SVC**

A two-stage pipeline that uses YOLOv8m as a feature extractor with PCA compression and SVM classification, achieving **87.52% accuracy** on the PlantWildV2 in-the-wild benchmark. Includes a novel treatment-based label engineering strategy that consolidates 115 disease classes into 11 actionable treatment categories.

## Key Results

| Pipeline | Accuracy | F1-Macro |
|----------|----------|----------|
| **YOLOv8m + IPCA + SVC** | **87.52%** | **0.882** |
| EfficientNet-B0 + IPCA + SVC | 70.53% | 0.668 |
| ResNet50 + IPCA + SVC | 68.27% | 0.645 |

## Repository Structure

```
├── paper/                          # LaTeX source and figures
│   ├── main.tex                    # Paper manuscript
│   ├── biblio.bib                  # Bibliography
│   └── plots/                      # All figures (9 files)
│
├── src/                            # Reproducible pipeline scripts
│   ├── preprocessing.py            # Feature extraction + PCA compression
│   ├── training_and_evaluation.py  # SVC training + grid search evaluation
│   ├── plot_results.py             # Result visualization
│   ├── generate_paper_plots.py     # Paper-quality figure generation
│   └── generate_per_class_report.py# Per-class classification metrics
│
└── notebooks/                      # Original Colab experiment notebooks
    ├── train_mlc_visual.py         # Classifier comparison (XGBoost, RF, MLP, SVC)
    ├── final_exp_plantwild_cusvm.py# Champion model deep-dive analysis
    └── working_code_of_yolov8.py   # Full YOLOv8 training + evaluation pipeline
```

## Reproducing the Experiments

### Prerequisites

```
Python 3.8+
PyTorch, torchvision
ultralytics (YOLOv8)
scikit-learn
xgboost
pandas, numpy, matplotlib, seaborn
timm (EfficientNet)
```

### Dataset

We use the [PlantWildV2](https://github.com/tqwei05/MVPDR) dataset by [Wei et al. (2024)](https://arxiv.org/abs/2412.13997). Download the dataset and update the paths in the configuration sections of each script.

### Pipeline

1. **Feature Extraction & PCA**: Run `src/preprocessing.py` to extract features from all three backbones (YOLOv8m, ResNet50, EfficientNet-B0) and apply dimensionality reduction.
2. **Training & Evaluation**: Run `src/training_and_evaluation.py` to train classifiers and generate performance metrics.
3. **Visualization**: Run `src/generate_paper_plots.py` to generate all paper figures.

## Method Overview

1. **Feature Extraction**: YOLOv8m backbone extracts spatially-aware activation maps from the final C2f block
2. **Dimensionality Reduction**: Incremental PCA compresses features to 100 dimensions (acts as regularizer)
3. **Classification**: SVC with RBF kernel (`C=1, γ=scale, class_weight=balanced`)
4. **Label Engineering**: 115 fine-grained classes → 11 treatment-based categories validated by a plant pathology expert

## Citation

Coming after acceptance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
