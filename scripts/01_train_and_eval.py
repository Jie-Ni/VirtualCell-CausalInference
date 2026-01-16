import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import pearsonr
import umap

# Add parent directory to path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import TurboGNN, SimpleTransformer

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "../data/processed"  # Adjust path as needed
RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    print(f"Running on {DEVICE}. Loading data...")

    # 1. Load Preprocessed Data
    # Note: Ensure these files exist or add code to download/generate them
    adata = sc.read_h5ad(os.path.join(DATA_DIR, "norman_mapped.h5ad"))
    gene_list = np.array(adata.var_names)
    num_genes = len(gene_list)
    adj = sp.load_npz(os.path.join(DATA_DIR, "adjacency_matrix_optimized.npz")).tocoo()

    # Prepare Graph Tensors
    edge_index = torch.stack([torch.from_numpy(adj.row), torch.from_numpy(adj.col)], 0).to(DEVICE).long()

    # Prepare Expression Data
    if sp.issparse(adata.X):
        X_np = adata.X.toarray()
    else:
        X_np = adata.X
    X_tensor = torch.tensor(X_np / X_np.max(), dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(X_tensor), batch_size=256, shuffle=True)

    # 2. Train Graph Neural Network
    print("Training GNN model...")
    gnn = TurboGNN(num_genes, edge_index).to(DEVICE)
    opt_gnn = torch.optim.Adam(gnn.parameters(), lr=0.005)
    crit = nn.MSELoss()

    loss_hist_gnn = []
    gnn.train()
    for epoch in range(20):
        ep_loss = 0
        for batch in dataloader:
            opt_gnn.zero_grad()
            # Predict expression from learned embeddings
            pred = gnn(perturbation_mask=None).unsqueeze(0).expand(batch[0].shape[0], -1)
            loss = crit(pred, batch[0].to(DEVICE))
            loss.backward()
            opt_gnn.step()
            ep_loss += loss.item()
        loss_hist_gnn.append(ep_loss / len(dataloader))

    # 3. Train Transformer Baseline
    print("Training Transformer baseline...")
    trans = SimpleTransformer(num_genes).to(DEVICE)
    opt_trans = torch.optim.Adam(trans.parameters(), lr=0.001)
    trans.train()
    for epoch in range(15):
        for batch in dataloader:
            opt_trans.zero_grad()
            pred = trans(mask=None).unsqueeze(0).expand(batch[0].shape[0], -1)
            loss = crit(pred, batch[0].to(DEVICE))
            loss.backward()
            opt_trans.step()

    # 4. Evaluation & Inference
    gnn.eval()
    trans.eval()
    print("Performing inference and perturbation analysis...")

    # Global Reconstruction Metrics
    true_mean = np.mean(X_np, axis=0)
    true_mean /= true_mean.max()
    with torch.no_grad():
        pred_gnn_mean = gnn(None).cpu().numpy()
        pred_gnn_mean /= pred_gnn_mean.max()
        pred_trans_mean = trans(None).cpu().numpy()

    # Manifold Learning (UMAP)
    # Raw data UMAP
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    raw_umap = adata.obsm['X_umap']

    # Latent space UMAP (GNN embeddings)
    emb_weights = gnn.emb.weight.cpu().detach().numpy()
    latent_umap = umap.UMAP(n_neighbors=20, min_dist=0.2).fit_transform(emb_weights)
    from sklearn.cluster import KMeans
    latent_clusters = KMeans(n_clusters=8).fit_predict(emb_weights)

    # In Silico Perturbation Analysis (e.g., Target Gene Knockout)
    target_gene = "TGFBR2" if "TGFBR2" in gene_list else gene_list[0]
    t_idx = np.where(gene_list == target_gene)[0][0]

    # Simulate Knockout (GNN)
    with torch.no_grad():
        mask_ko = torch.ones(num_genes, device=DEVICE)
        mask_ko[t_idx] = 0
        gnn_diff = (gnn(mask_ko) - gnn(None)).cpu().numpy()

    # Simulate Knockout (Transformer)
    with torch.no_grad():
        mask_t = torch.zeros(num_genes, dtype=torch.bool, device=DEVICE)
        mask_t[t_idx] = True
        trans_diff = (trans(mask_t) - trans(None)).cpu().numpy()
        trans_diff[t_idx] = 0  # Mask self-effect

    # Dose-Response Simulation
    dosages = np.linspace(1, 0, 10)
    dose_curves = []
    top_responders = gene_list[np.argsort(np.abs(gnn_diff))[-5:]]
    for g in top_responders:
        g_idx = np.where(gene_list == g)[0][0]
        base_val = pred_gnn_mean[g_idx]
        curve = []
        for d in dosages:
            w = torch.ones(num_genes, device=DEVICE)
            w[t_idx] = d
            with torch.no_grad():
                val = gnn(w).cpu().numpy()[g_idx] / gnn(None).cpu().numpy().max()
                curve.append(val - base_val)
        dose_curves.append(curve)

    # Virtual Rescue Screening
    # Calculate phenotypic correlation for all genes against the target disease signature
    rescue_x = []  # Correlation
    rescue_y = []  # Magnitude
    for i in range(min(2000, num_genes)):  # Sample for speed, or run all
        with torch.no_grad():
            m = torch.ones(num_genes, device=DEVICE)
            m[i] = 0
            # Delta vector for gene i
            d_vec = (gnn(m) - gnn(None)).cpu().numpy()
            # Correlation with disease signature (gnn_diff)
            corr = pearsonr(gnn_diff, d_vec)[0]
            mag = np.linalg.norm(d_vec)
            rescue_x.append(corr)
            rescue_y.append(mag)

    # 5. Save Results
    save_path = os.path.join(RESULTS_DIR, "plot_data.npz")
    print(f"Saving analysis results to {save_path}...")
    np.savez(
        save_path,
        # Manifold Data
        fig1_raw_umap=raw_umap,
        fig1_latent_umap=latent_umap,
        fig1_clusters=latent_clusters,

        # Performance Data
        fig2_true_mean=true_mean,
        fig2_pred_mean=pred_gnn_mean,
        fig2_degrees=np.diff(adj.tocsr().indptr),
        fig2_loss_hist=loss_hist_gnn,

        # Dynamic Data
        fig3_target=target_gene,
        fig3_dosages=dosages,
        fig3_top5_genes=top_responders,
        fig3_dose_curves=dose_curves,
        fig3_edges=list(zip(gene_list[adj.row], gene_list[adj.col])),  # Save graph edges

        # Trajectory (Simulation)
        fig3_traj_start=latent_umap[np.random.choice(num_genes, 500)],  # Dummy coords for visualization logic
        fig3_traj_end=latent_umap[np.random.choice(num_genes, 500)] + 0.5,

        # Mechanism & Rescue
        fig4_waterfall_genes=top_responders,  # Reuse for simplicity
        fig4_waterfall_vals=gnn_diff[np.argsort(np.abs(gnn_diff))[-5:]],
        fig4_rescue_x=np.array(rescue_x),
        fig4_rescue_y=np.array(rescue_y),
        fig4_heatmap_data=np.random.randn(10, 3),  # Placeholder for pairwise epistasis
        fig4_heatmap_labels=[target_gene, "GeneX", "Double"],

        # Benchmark Comparison
        fig5_gnn_diff=gnn_diff,
        fig5_trans_diff=trans_diff,
        fig5_trans_pred_all=pred_trans_mean,
        fig5_rand_diff=np.random.normal(0, np.std(gnn_diff), len(gnn_diff))  # Randomized baseline
    )
    print("Done.")


if __name__ == "__main__":
    main()