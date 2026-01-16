import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
from scipy.stats import pearsonr
import os
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects

# Configuration
RESULTS_DIR = "../results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DATA_PATH = os.path.join(RESULTS_DIR, "plot_data.npz")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Global Style
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 14
})


def add_text_halo(text_objects):
    for t in text_objects.values():
        t.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])


def main():
    if not os.path.exists(DATA_PATH):
        print("Data file not found. Run training script first.")
        return
    D = np.load(DATA_PATH, allow_pickle=True)
    print("Generating visualization plots...")

    # --- Plot 1: Framework and Manifold ---
    # Visualizes the raw data distribution and the learned latent structure.
    fig1 = plt.figure(figsize=(20, 16), constrained_layout=True)
    gs1 = gridspec.GridSpec(2, 2, height_ratios=[0.8, 1], figure=fig1)

    # Schematic placeholder
    ax1_a = fig1.add_subplot(gs1[0, :])
    ax1_a.text(0.5, 0.5, "Schematic Diagram (Model Architecture)", ha='center', fontsize=20)
    ax1_a.axis('off')

    # Latent space visualization
    ax1_c = fig1.add_subplot(gs1[1, 1])
    lat_umap = D['fig1_latent_umap']
    clusters = D['fig1_clusters']
    scatter = ax1_c.scatter(lat_umap[:, 0], lat_umap[:, 1], c=clusters, cmap='tab10', s=15, alpha=0.8)
    ax1_c.set_title("Learned Latent Space")
    ax1_c.axis('off')

    # (Additional subplots omitted for brevity, logic follows same pattern...)
    fig1.savefig(os.path.join(FIGURES_DIR, "Figure1_Framework.png"))
    plt.close(fig1)

    # --- Plot 2: Dynamics and Trajectory ---
    # Visualizes dose-response curves and vector fields.
    fig3 = plt.figure(figsize=(20, 16), constrained_layout=True)
    gs3 = gridspec.GridSpec(2, 2, height_ratios=[0.8, 1.2], figure=fig3)

    # Dose response
    ax3_a = fig3.add_subplot(gs3[0, :])
    dosages = D['fig3_dosages']
    curves = D['fig3_dose_curves']
    for i, c in enumerate(curves):
        ax3_a.plot(dosages, c, 'o-', lw=3)
    ax3_a.set_title("In Silico Dose-Response Simulation")
    ax3_a.invert_xaxis()

    fig3.savefig(os.path.join(FIGURES_DIR, "Figure3_Dynamics.png"))
    plt.close(fig3)

    # --- Plot 3: Benchmark and Paradox ---
    # Compares reconstruction vs causal inference (GNN vs Transformer).
    fig5 = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs5 = gridspec.GridSpec(2, 2, figure=fig5)

    # Causal Prediction scatter
    ax5_b = fig5.add_subplot(gs5[0, 1])
    dg = D['fig5_gnn_diff']
    dt = D['fig5_trans_diff']
    ax5_b.scatter(dg, dt, c='gray', alpha=0.3, label='Genome-wide')
    ax5_b.set_xlabel("Structure-based Prediction")
    ax5_b.set_ylabel("Sequence-based Prediction")
    ax5_b.set_title("Generalization to Unseen Perturbations")

    fig5.savefig(os.path.join(FIGURES_DIR, "Figure5_Benchmark.png"))
    plt.close(fig5)

    print(f"All figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()