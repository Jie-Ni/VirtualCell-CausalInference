Towards a Causal Virtual Cell: Graph-Structured Priors Decouple Inference from Representation
This repository contains the official PyTorch implementation and data reproduction pipeline for the manuscript:
Towards a Causal Virtual Cell: Graph-Structured Priors Decouple Inference from Representation
Submitted to Nature Communications
🧬 Overview
Current foundation models in single-cell biology (e.g., scGPT, Geneformer) excel at reconstructing gene expression profiles but often fail to predict the effects of unseen perturbations. We term this phenomenon the "Accuracy-Causality Paradox".
This project implements TurboGNN, a Knowledge-Primed Graph Neural Network that integrates biological structural priors (PPI and GO networks) into the latent space. By constraining message passing within a valid biological topology, our framework decouples causal inference from representation learning, enabling:
Robust Generalization: Predicting phenotypic shifts for gene knockouts not seen during training.
In Silico Dynamics: Simulating dose-response curves and cell state trajectories.
Virtual Drug Screening: Identifying therapeutic rescue targets and epistatic interactions ab initio.
📂 Repository Structure
.
├── data/                   # Data storage
│   ├── download_data.sh    # Script to fetch datasets from Zenodo
│   ├── raw/                # Raw input files (auto-generated)
│   └── processed/          # Processed adjacency matrices & expression data
├── src/                    # Model source code
│   ├── __init__.py
│   └── models.py           # Implementation of TurboGNN and Transformer baseline
├── scripts/                # Reproduction pipelines
│   ├── 01_train_and_eval.py    # Main training & inference loop
│   ├── 02_visualize_results.py # Generates Figures 1-7
│   └── 03_make_tables.py       # Generates Tables 1-3
├── results/                # Output directory
│   ├── figures/            # Generated plots (.png/.tif)
│   └── tables/             # Generated statistical tables (.csv)
├── requirements.txt        # Python dependencies
└── README.md               # This file


🛠️ Installation
Prerequisites
Linux, macOS, or Windows
Python 3.8+
NVIDIA GPU (Recommended for Transformer training, typically requires 8GB+ VRAM)
Step 1: Clone the repository
git clone [https://github.com/YourUsername/Causal-Virtual-Cell.git](https://github.com/YourUsername/Causal-Virtual-Cell.git)
cd Causal-Virtual-Cell


Step 2: Set up the environment
We recommend using Conda to manage dependencies:
conda create -n virtualcell python=3.9
conda activate virtualcell
pip install -r requirements.txt


⬇️ Data Preparation
The processed datasets (Perturb-seq expression matrices and Knowledge Graph adjacency matrices) are hosted on Zenodo due to size constraints.
To download the data, run the provided script:
cd data
bash download_data.sh
cd ..


Dataset DOI: 10.5281/zenodo.18271659
Note: The script automatically fetches norman_mapped.h5ad and adjacency_matrix_optimized.npz and places them in data/processed/.
🚀 Reproduction Instructions
To reproduce the results, figures, and tables presented in the manuscript, please execute the scripts in the following order.
1. Training & Inference
This script trains both the TurboGNN (Ours) and the Sequence Transformer (Baseline), performs in silico perturbations, and computes manifold projections.
python scripts/01_train_and_eval.py


Output: results/plot_data.npz (Contains all numerical results and embeddings).
Runtime: ~5-20 minutes depending on GPU.
2. Generate Figures
This script reads the analysis results and generates Figures 1 through 7 as seen in the paper.
python scripts/02_visualize_results.py


Output: Images saved in results/figures/.
Figure1_Framework.png: Model architecture & Latent manifold.
Figure2_Performance.png: Reconstruction & Robustness metrics.
Figure3_Dynamics.png: Dose-response & Trajectories.
Figure4_Mechanism.png: Virtual rescue & Epistasis.
Figure5_Benchmark.png: The Accuracy-Causality Paradox.
Figure6_Interpretation.png: Pathway enrichment & Attention.
Figure7_Diagnostics.png: Model diagnostics & Extended analysis.
3. Generate Tables
This script computes the statistical metrics (Pearson R, Jaccard Index, etc.) and generates the comparison tables.
python scripts/03_make_tables.py


Output: CSV files saved in results/tables/.
Table1_Reconstruction.csv: Performance metrics.
Table2_Causality.csv: Quantification of OOD generalization.
Table3_Drug_Candidates.csv: Top hits from virtual screening.
💻 Hardware Requirements
The code has been tested on the following configuration:
OS: Ubuntu 20.04 LTS
CPU: AMD EPYC / Intel Xeon (16 cores)
GPU: NVIDIA A100 (80GB) or RTX 3090 (24GB) - Code supports CPU-only execution but will be slower.
RAM: 32GB+
📄 Citation
If you find this code or framework useful in your research, please cite our manuscript:
@article{Ni2025CausalVirtualCell,
  title={Towards a Causal Virtual Cell: Graph-Structured Priors Decouple Inference from Representation},
  author={Ni, Jie and [Co-Authors]},
  journal={Submitted to Nature Communications},
  year={2025}
}


⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.
📧 Contact
For any questions regarding the code or data, please open an issue or contact:
Jie Ni Email: njie@seu.edu.cn
Southeast university
