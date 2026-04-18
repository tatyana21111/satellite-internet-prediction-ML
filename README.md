# Predicting Internet Connectivity from Satellite Imagery

Predicting internet connectivity access across India using satellite imagery and geospatial foundation models, with a geographic out-of-distribution evaluation on Northeast India.

## Problem

Reliable maps of internet access are critical for infrastructure planning, yet ground-truth connectivity data is sparse in developing regions. This project investigates whether satellite imagery -- processed through geospatial foundation models (Prithvi-EO-2.0) and classical ML baselines -- can predict internet connectivity at the sub-national level. A key challenge is **geographic distribution shift**: models are trained on most of India but evaluated on the geographically and socioeconomically distinct Northeast region.

## Data

| Source | Description |
|--------|-------------|
| [Harmonized Landsat Sentinel-2 (HLS)](https://hls.gsfc.nasa.gov/) | 6-band satellite imagery, 224x224 px patches at 30 m resolution (6,970 patches) |
| [Ookla Speedtest Intelligence](https://www.ookla.com/ookla-for-good/open-data) | Internet speed-test tile data (Q1 2019), used to construct connectivity labels |
| Engineered features | Nighttime lights (VIIRS), population density (WorldPop), road density (OSM), elevation/slope (SRTM) |

**Label construction:** Two aggregate regression targets are derived from Ookla sub-tiles per patch:
- `coverage_fraction` -- spatial coverage (proportion of zoom-16 tiles with data)
- `log_mean_tests` -- log-transformed mean test density

Binary connectivity labels are also used for baseline comparison.

**Train/test split:** Geographic hold-out -- Northeast India (lon > 89, lat > 21) is reserved as the test set; remaining India is split 80/20 for train/val with stratification.

## Methods

The project compares three tiers of models:

### Tier 1: Classical Baselines (Notebooks 02-03)
- **XGBoost** on engineered socioeconomic features (nightlights, population, roads, elevation)
- **MOSAIKS** random kitchen-sink features + Ridge regression
- **ResNet-18** (ImageNet-pretrained) + logistic/linear head
- **DINOv2** (ViT-L/14, frozen) + logistic/linear head

### Tier 2: Geospatial Foundation Model — Head-Capacity Ablation (Notebook 04)
- **Prithvi-EO-2.0-300M** (frozen encoder, 1024-d embeddings)
  - Linear probe: `LayerNorm(1024) -> Dropout(0.1) -> Linear(1024, 1)`
  - MLP head: `LayerNorm(1024) -> Linear(1024, 256) -> GELU -> Dropout(0.2) -> Linear(256, 128) -> GELU -> Dropout(0.2) -> Linear(128, 1)`

### Tier 3: Feature Fusion (Notebook 05)
- **XGBoost + Prithvi embeddings** -- combines engineered socioeconomic features with frozen Prithvi-EO embeddings, with ablation over feature subsets

### Qualitative Analysis (Notebook 06)
- Spatial error maps, regional breakdowns, satellite overlay case studies

### Data Overview (Notebook 07)
- Target distribution histograms, binary label imbalance, sample statistics table, spatial distribution map of `log_mean_tests` across India

### Location Embedding Ablation (Notebook 08)
- **Prithvi-EO-2.0-300M-TL** with location & temporal embeddings — Linear, MLP, and XGBoost heads compared against base Prithvi models from NB04/05

## Notebooks

Run notebooks in order on **Google Colab with GPU** (T4 minimum). NB01 must be run first to generate all data; NB02-08 consume the outputs.

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_data_exploration.ipynb` | Download Ookla data, build patch grid, extract stratification features, sample patches, export HLS imagery, compute aggregate targets, and save train/val/test splits |
| 02 | `02_binary_baselines.ipynb` | Five binary classification baselines (XGBoost, MOSAIKS, ResNet-18, DINOv2, Prithvi linear) |
| 03 | `03_aggregate_baselines.ipynb` | MOSAIKS, ResNet-18, DINOv2 retrained on aggregate regression targets |
| 04 | `04_prithvi_head_ablation.ipynb` | Prithvi-EO-2.0 frozen encoder — Linear probe vs MLP head (combined ablation with train/val loss plots) |
| 05 | `05_xgboost_prithvi_fusion.ipynb` | XGBoost combining engineered features + Prithvi embeddings, with ablation |
| 06 | `06_qualitative_analysis.ipynb` | Spatial error analysis, regional summaries, satellite overlay case studies |
| 07 | `07_sample_statistics.ipynb` | Target histograms, binary label imbalance, sample statistics table, spatial India map of `log_mean_tests` |
| 08 | `08_prithvi_location_embedding.ipynb` | Prithvi-EO-2.0-300M-TL with location/temporal embeddings — Linear, MLP, XGBoost heads vs base models |

### NB01 outputs

NB01 produces the following files that all downstream notebooks depend on:

| File | Location | Persistence |
|------|----------|-------------|
| ~6,970 GeoTIFF patches (224×224×6 HLS bands) | `data/raw/patches_2019/*.tif` | Synced to Google Drive at end of Step 7 |
| Sampled patch metadata + binary labels | `data/processed/sampled_patches.csv` | Committed to git |
| Metadata + aggregate targets + train/val/test splits | `data/processed/patches_with_splits.csv` | Committed to git |

**Important — Google Drive persistence:**
- **GeoTIFF patches** (~2 GB): Too large for git. NB01 Step 7b copies them to `Google Drive → patches_2019/`. NB02-08 sync them back at startup (Cell 0.4).
- **Prithvi embeddings** (~80 MB): Whichever of NB04/05/08 runs first extracts frozen 1024-d embeddings and caches them to `Google Drive → prithvi_embeddings/`. Subsequent notebooks load from Drive, avoiding redundant extraction and ensuring identical embeddings across experiments.

## Evaluation

**Primary metrics** (regression): MAE, RMSE, Spearman rho on continuous targets.

**Secondary metrics** (binary task): PR-AUC, ROC-AUC, F1 at val-optimal threshold -- regression predictions are binarised and compared against binary ground-truth labels.

All metrics are reported on the **Northeast India geographic hold-out test set**.

## Repository Structure

```
satellite-internet-prediction/
├── notebooks/          # Jupyter notebooks (01-08), run on Colab
├── data/
│   ├── raw/            # GeoTIFF patches, Ookla CSV (not committed)
│   └── processed/      # Patch metadata CSV (not committed)
├── outputs/
│   ├── results/        # Metrics CSVs per model
│   ├── models/         # Trained model artifacts (.pkl, .json)
│   ├── figures/        # Generated plots
│   └── features/       # Cached Prithvi embeddings (not committed)
├── .gitignore
└── README.md
```

## Reproducibility

### First-time setup (run once)
1. Open `01_data_exploration.ipynb` in Google Colab (GPU runtime).
2. Run all cells. Steps 1-7 take ~2-3 hours (GEE exports + HLS patch downloads).
3. At the end of Step 7, patches are automatically saved to Google Drive (`patches_2019/`).
4. Step 8 computes aggregate targets and saves `patches_with_splits.csv` (committed to git).

### Running NB02-08
1. Open any notebook in Google Colab (GPU runtime).
2. Cell 0.1 installs dependencies. For NB04-06 and NB08, the runtime restarts after installing `terratorch` — re-run from Cell 0.2 after restart.
3. Cell 0.2 clones this repository.
4. Cell 0.3 syncs the ~7K GeoTIFF patches from Google Drive into `data/raw/patches_2019/`.
5. Step 2 loads `patches_with_splits.csv` for metadata, labels, and pre-computed splits.
6. Run all remaining cells sequentially. Results are saved to `outputs/`.

**Seeds:** All notebooks set `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)` at the top for reproducibility.

## Key References

- Jakubik, J. et al. (2024). *Prithvi-EO-2.0.* IBM/NASA geospatial foundation model.
- Ookla for Good. (2019). *Speedtest Intelligence open data.*
- Rolf, E. et al. (2021). *A generalizable and accessible approach to machine learning with global satellite imagery.* Nature Communications.
