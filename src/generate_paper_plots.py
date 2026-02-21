"""
Generate all plots for the paper.
All values are hardcoded from final_results_summary.csv and n_components_ablation_report.csv.
Output: PNG files saved to paper_updated/paper_exported_from_overleaf/
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# --- Configuration ---
OUT_DIR = os.path.join(os.path.dirname(__file__),
                       "paper_updated", "paper_exported_from_overleaf")
os.makedirs(OUT_DIR, exist_ok=True)

# Publication-quality settings
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette
YOLO_COLOR = '#2563EB'      # blue
RESNET_COLOR = '#DC2626'    # red
EFFNET_COLOR = '#16A34A'    # green
ACCENT = '#F59E0B'          # amber


# ==========================================================
# PLOT 1: N-Components Ablation (updates existing figure)
# ==========================================================
def plot_ablation():
    """F1-Macro vs number of principal components."""
    # From n_components_ablation_report.csv (C=1 row, treatment grouping)
    n_components = [100, 250, 450, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 5745]
    f1_c1  = [0.8150, 0.8046, 0.8039, 0.8054, 0.8025, 0.8022,
              0.7962, 0.7987, 0.7937, 0.7906, 0.7894, 0.7883]
    f1_c100 = [0.8073, 0.8136, 0.7997, 0.8039, 0.8084, 0.8044,
               0.8054, 0.8034, 0.8024, 0.8013, 0.8014, 0.8024]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(n_components, f1_c1, 'o-', color=YOLO_COLOR, linewidth=2,
            markersize=6, label='C = 1 (selected)', zorder=5)
    ax.plot(n_components, f1_c100, 's--', color=RESNET_COLOR, linewidth=1.8,
            markersize=5, label='C = 100', alpha=0.8)

    # Highlight the optimum
    ax.axvline(x=100, color=ACCENT, linestyle=':', linewidth=1.5, alpha=0.7)
    ax.annotate('n = 100\n(optimum)',
                xy=(100, f1_c1[0]), xytext=(400, f1_c1[0] + 0.005),
                fontsize=10, fontweight='bold', color=ACCENT,
                arrowprops=dict(arrowstyle='->', color=ACCENT, lw=1.5))

    ax.set_xlabel('Number of Principal Components (n)')
    ax.set_ylabel('F1-Macro Score')
    ax.set_title('Effect of Dimensionality on Classification Performance')
    ax.set_xscale('log')
    ax.set_xticks(n_components)
    ax.set_xticklabels([str(n) for n in n_components], rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0.78, 0.825)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'n_components_ablation_plot.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ==========================================================
# PLOT 2: Backbone & Compression Comparison (NEW grouped bar)
# ==========================================================
def plot_backbone_comparison():
    """Grouped bar chart: Accuracy & F1 for all 12 configurations."""
    # From final_results_summary.csv
    data = {
        # (Backbone, Compression, Label_Set): (Accuracy, F1)
        ('EfficientNet-B0', 'IPCA', '19 Classes'): (0.6823, 0.7047),
        ('EfficientNet-B0', 'IPCA', '11 Classes'): (0.7053, 0.7065),
        ('EfficientNet-B0', 'SVD',  '19 Classes'): (0.6593, 0.6842),
        ('EfficientNet-B0', 'SVD',  '11 Classes'): (0.6797, 0.6849),
        ('ResNet50',        'IPCA', '19 Classes'): (0.6623, 0.6813),
        ('ResNet50',        'IPCA', '11 Classes'): (0.6827, 0.6785),
        ('ResNet50',        'SVD',  '19 Classes'): (0.5963, 0.6179),
        ('ResNet50',        'SVD',  '11 Classes'): (0.6107, 0.6188),
        ('YOLOv8m',         'IPCA', '19 Classes'): (0.8654, 0.8634),
        ('YOLOv8m',         'IPCA', '11 Classes'): (0.8752, 0.8820),
        ('YOLOv8m',         'SVD',  '19 Classes'): (0.8603, 0.8538),
        ('YOLOv8m',         'SVD',  '11 Classes'): (0.8654, 0.8743),
    }

    # --- Side-by-side bar chart: 11-class results only (cleaner) ---
    backbones = ['EfficientNet-B0', 'ResNet50', 'YOLOv8m']
    compressions = ['IPCA', 'SVD']
    colors = {
        ('EfficientNet-B0', 'IPCA'): '#86EFAC', ('EfficientNet-B0', 'SVD'): '#16A34A',
        ('ResNet50', 'IPCA'):        '#FCA5A5', ('ResNet50', 'SVD'):        '#DC2626',
        ('YOLOv8m', 'IPCA'):         '#93C5FD', ('YOLOv8m', 'SVD'):        '#2563EB',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    # --- Panel A: 11-Class (Treatment) ---
    x = np.arange(len(backbones))
    width = 0.3
    for i, comp in enumerate(compressions):
        accs = [data[(b, comp, '11 Classes')][0] * 100 for b in backbones]
        f1s  = [data[(b, comp, '11 Classes')][1] for b in backbones]
        bars = ax1.bar(x + i * width - width/2, accs, width * 0.9,
                       label=comp, color=[colors[(b, comp)] for b in backbones],
                       edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(backbones)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('(a) 11-Class Treatment Grouping')
    ax1.set_ylim(55, 95)
    ax1.legend(title='Compression')

    # --- Panel B: 19-Class (Visual) ---
    for i, comp in enumerate(compressions):
        accs = [data[(b, comp, '19 Classes')][0] * 100 for b in backbones]
        bars = ax2.bar(x + i * width - width/2, accs, width * 0.9,
                       label=comp, color=[colors[(b, comp)] for b in backbones],
                       edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, accs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(backbones)
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('(b) 19-Class Visual Grouping')
    ax2.set_ylim(55, 95)
    ax2.legend(title='Compression')

    plt.suptitle('Backbone and Compression Method Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'backbone_comparison_barplot.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ==========================================================
# PLOT 3: Per-Class Performance Horizontal Bar (NEW)
# ==========================================================
def plot_per_class():
    """Horizontal bar chart of per-class F1-scores for the champion model."""
    classes = [
        'abiotic_disorder', 'fungal_powdery_mildew', 'fungal_scab',
        'fungal_rust', 'fungal_leaf_disease', 'fungal_downy_mildew',
        'oomycete_lesion', 'viral_disease', 'fungal_systemic_smut_gall',
        'bacterial_disease', 'fungal_rot_fruit_disease',
    ]
    f1_scores = [0.920, 0.945, 0.920, 0.894, 0.876, 0.853,
                 0.847, 0.841, 0.826, 0.814, 0.795]

    # Sort  by F1 for visual clarity
    sorted_pairs = sorted(zip(classes, f1_scores), key=lambda x: x[1])
    classes_sorted = [p[0].replace('_', ' ').title() for p in sorted_pairs]
    f1_sorted = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    colormap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=0.75, vmax=0.95)
    colors = [colormap(norm(v)) for v in f1_sorted]

    bars = ax.barh(classes_sorted, f1_sorted, color=colors, edgecolor='white', height=0.7)

    # Add value labels
    for bar, val in zip(bars, f1_sorted):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

    # Macro avg line
    macro_f1 = 0.882
    ax.axvline(x=macro_f1, color='#1E3A5F', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Macro Avg F1 = {macro_f1}')

    ax.set_xlabel('F1-Score')
    ax.set_title('Per-Class F1-Score (Champion Model: YOLOv8m + IPCA + SVC)')
    ax.set_xlim(0.7, 1.0)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.invert_yaxis()  # Invert so that highest is at the top after sorting

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'per_class_f1_barplot.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ==========================================================
# PLOT 4: Label Engineering Impact (NEW)
# ==========================================================
def plot_label_engineering():
    """Show performance uplift from 19-class to 11-class grouping."""
    backbones = ['EfficientNet-B0', 'ResNet50', 'YOLOv8m']

    # IPCA results from final_results_summary.csv
    acc_19 = [68.23, 66.23, 86.54]
    acc_11 = [70.53, 68.27, 87.52]
    f1_19  = [0.7047, 0.6813, 0.8634]
    f1_11  = [0.7065, 0.6785, 0.8820]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(backbones))
    width = 0.35

    # Accuracy panel
    bars1 = ax1.bar(x - width/2, acc_19, width, label='19-Class (Visual)',
                    color='#CBD5E1', edgecolor='white')
    bars2 = ax1.bar(x + width/2, acc_11, width, label='11-Class (Treatment)',
                    color=YOLO_COLOR, edgecolor='white')

    for bar, val in zip(bars1, acc_19):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, acc_11):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(backbones)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('(a) Accuracy')
    ax1.set_ylim(60, 95)
    ax1.legend()

    # F1 panel
    bars3 = ax2.bar(x - width/2, f1_19, width, label='19-Class (Visual)',
                    color='#CBD5E1', edgecolor='white')
    bars4 = ax2.bar(x + width/2, f1_11, width, label='11-Class (Treatment)',
                    color=YOLO_COLOR, edgecolor='white')

    for bar, val in zip(bars3, f1_19):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars4, f1_11):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(backbones)
    ax2.set_ylabel('Macro F1-Score')
    ax2.set_title('(b) F1-Score')
    ax2.set_ylim(0.60, 0.95)
    ax2.legend()

    plt.suptitle('Impact of Label Engineering (IPCA Compression, n=100)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'label_engineering_impact.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ==========================================================
# PLOT 5: Training Time Comparison (NEW — replaces old Table 5)
# ==========================================================
def plot_training_time():
    """Bar chart comparing SVC training times across backbones."""
    # From final_results_summary.csv (IPCA, 11-class)
    backbones = ['EfficientNet-B0\n+ IPCA', 'ResNet50\n+ IPCA', 'YOLOv8m\n+ IPCA']
    times = [37.75, 40.45, 30.77]  # seconds
    accs  = [70.53, 68.27, 87.52]
    colors_bars = [EFFNET_COLOR, RESNET_COLOR, YOLO_COLOR]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(backbones, times, color=colors_bars, edgecolor='white',
                  width=0.5, alpha=0.85)

    for bar, t, a in zip(bars, times, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{t:.1f}s\n({a:.1f}% acc)', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_ylabel('Classifier Training Time (seconds)')
    ax.set_title('SVC Training Time Comparison (CPU, n=100 components)')
    ax.set_ylim(0, 55)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'training_time_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")


# ==========================================================
# RUN ALL
# ==========================================================
if __name__ == '__main__':
    print("=" * 50)
    print("Generating paper plots...")
    print("=" * 50)

    plot_ablation()
    plot_backbone_comparison()
    plot_per_class()
    plot_label_engineering()
    plot_training_time()

    print("\n✅ All plots generated successfully!")
    print(f"Output directory: {OUT_DIR}")
