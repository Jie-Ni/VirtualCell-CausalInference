import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os

RESULTS_DIR = "../results"
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
DATA_PATH = os.path.join(RESULTS_DIR, "plot_data.npz")
os.makedirs(TABLES_DIR, exist_ok=True)


def main():
    if not os.path.exists(DATA_PATH):
        print("Data file not found.")
        return
    D = np.load(DATA_PATH, allow_pickle=True)
    print("Generating statistical tables...")

    # Table 1: Reconstruction Metrics
    true_expr = D['fig2_true_mean']
    gnn_pred = D['fig2_pred_mean']
    trans_pred = D['fig5_trans_pred_all']

    # Calculate metrics (Dummy function for brevity, real implementation uses sklearn/scipy)
    r_gnn = pearsonr(true_expr, gnn_pred)[0]
    r_trans = pearsonr(true_expr, trans_pred)[0]

    df1 = pd.DataFrame({
        "Method": ["Sequence Transformer", "TurboGNN"],
        "Pearson R": [r_trans, r_gnn]
    })
    df1.to_csv(os.path.join(TABLES_DIR, "Table1_Metrics.csv"), index=False)

    print(f"Tables saved to {TABLES_DIR}")


if __name__ == "__main__":
    main()