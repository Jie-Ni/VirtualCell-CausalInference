#!/bin/bash

# ==============================================================================
# Script to download datasets for the Virtual Cell Causal Inference Project
# DOI: 10.5281/zenodo.18271659
# ==============================================================================
# Usage:
#   cd data
#   bash download_data.sh
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}[INFO] Starting data download process...${NC}"

# 1. Create directory structure if it doesn't exist
# ------------------------------------------------------------------------------
echo -e "${BLUE}[INFO] Creating directory structure...${NC}"
mkdir -p raw
mkdir -p processed

# 2. Define Zenodo URLs
# ------------------------------------------------------------------------------
# Based on DOI: 10.5281/zenodo.18271659
# Record ID: 18271659
# ------------------------------------------------------------------------------

# Link for norman_mapped.h5ad
NORMAN_DATA_URL="https://zenodo.org/record/18271659/files/norman_mapped.h5ad?download=1"

# Link for adjacency_matrix_optimized.npz
ADJ_MATRIX_URL="https://zenodo.org/record/18271659/files/adjacency_matrix_optimized.npz?download=1"

# 3. Download the Main Expression Data
# ------------------------------------------------------------------------------
if [ -f "processed/norman_mapped.h5ad" ]; then
    echo -e "${GREEN}[SKIP] norman_mapped.h5ad already exists.${NC}"
else
    echo -e "${BLUE}[DOWNLOADING] Norman et al. transcriptomic data...${NC}"
    # Try wget first, fall back to curl
    if command -v wget &> /dev/null; then
        wget -O processed/norman_mapped.h5ad "$NORMAN_DATA_URL"
    else
        curl -L -o processed/norman_mapped.h5ad "$NORMAN_DATA_URL"
    fi
    
    # Check if download was actually successful (size > 0)
    if [ -s "processed/norman_mapped.h5ad" ]; then
        echo -e "${GREEN}[SUCCESS] Downloaded norman_mapped.h5ad${NC}"
    else
        echo -e "${RED}[ERROR] Failed to download norman_mapped.h5ad. Please check the Zenodo link.${NC}"
        exit 1
    fi
fi

# 4. Download the Graph Structure (Adjacency Matrix)
# ------------------------------------------------------------------------------
if [ -f "processed/adjacency_matrix_optimized.npz" ]; then
    echo -e "${GREEN}[SKIP] adjacency_matrix_optimized.npz already exists.${NC}"
else
    echo -e "${BLUE}[DOWNLOADING] Knowledge Graph Adjacency Matrix...${NC}"
    if command -v wget &> /dev/null; then
        wget -O processed/adjacency_matrix_optimized.npz "$ADJ_MATRIX_URL"
    else
        curl -L -o processed/adjacency_matrix_optimized.npz "$ADJ_MATRIX_URL"
    fi
    
    if [ -s "processed/adjacency_matrix_optimized.npz" ]; then
        echo -e "${GREEN}[SUCCESS] Downloaded adjacency_matrix_optimized.npz${NC}"
    else
        echo -e "${RED}[ERROR] Failed to download adjacency_matrix_optimized.npz. Please check the Zenodo link.${NC}"
        exit 1
    fi
fi

# 5. Final Check
# ------------------------------------------------------------------------------
echo -e "${BLUE}[INFO] Verifying files...${NC}"
ls -lh processed/

echo -e "${GREEN}======================================================${NC}"
echo -e "${GREEN}  All data downloaded successfully from Zenodo!       ${NC}"
echo -e "${GREEN}======================================================${NC}"