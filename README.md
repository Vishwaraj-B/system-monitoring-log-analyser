# System Monitoring & Log Metric Analyser

> A complete ML pipeline that ingests 4 million real supercomputer logs, engineers time-window features, and runs 11 models across classification, clustering, and regression to detect anomalies, predict failures, and monitor system health in real time.

---

## What It Does

This project processes the **BGL (BlueGene/L) supercomputer log dataset** — 4+ million raw log lines from a real IBM supercomputer — and builds an end-to-end intelligent monitoring system using machine learning. It answers three operational questions:

1. **Is this time window anomalous right now?** → Anomaly Detection (Classification)
2. **Will the next window have a failure?** → Failure Prediction (look-ahead classification + regression)
3. **What does the system health look like over time?** → Monitoring Dashboards

---

## Dataset

**BGL Structured Log Dataset** from Kaggle  
Source: `ayush2222/structured-bgl-logs-csv` (BlueGene/L supercomputer, Livermore National Lab)

- **4,188,776 log lines** from a real production supercomputer
- Each log has a `Label` column — `-` = normal, anything else = a specific failure type (KERNDTLB, KERNSTOR, etc.)
- Logs span years of operation with IBM's custom timestamp format (`YYYY-MM-DD-HH.MM.SS.ffffff`)
- ~10% anomalous, ~90% normal — a classic imbalanced real-world dataset

---

## How It Works

### Pipeline Overview

```
Raw CSV (4M rows)
    ↓
Preprocessing (timestamp parsing, null handling, sort by time)
    ↓
Feature Engineering (5-min time windows per node)
    ↓
Feature Extraction (event_count, fatal_count, fatal_ratio, etc.)
    ↓
EDA + Outlier Analysis (IQR detection, winsorization)
    ↓
┌─────────────────────────────────────────────┐
│  Classification   Clustering   Regression   │
│  KNN              DBSCAN        Lin. Reg.   │
│  Decision Tree    Spectral      XGBoost Reg │
│  LinearSVC                                  │
│  Random Forest                              │
│  XGBoost                                    │
└─────────────────────────────────────────────┘
    ↓
Final Comparison Table + ROC-AUC Curves
    ↓
System Monitoring Dashboards (time-series plots)
```

### Feature Engineering

Rather than using raw log lines, the pipeline groups logs into **5-minute time windows per node** and computes:

| Feature | What It Captures |
|---|---|
| `event_count` | Total log volume in the window |
| `unique_event_count` | How many distinct event types occurred |
| `fatal_count` | Number of FATAL-level events |
| `info_count` | Number of INFO-level events |
| `fatal_ratio` | fatal_count / event_count — the most predictive signal |

### Why Binary Labels

The original dataset has 10+ anomaly sub-types, but many have under 5 samples. All sub-types are merged into a single anomaly class (1 = anomaly, 0 = normal) for practical binary classification — which is exactly what a real monitoring system needs.

---

## Models Implemented

### Classification (Anomaly Detection)
| Model | Key Tuning | Notes |
|---|---|---|
| **KNN** | k ∈ {1,3,5,7,9,11,15} | Scaled features, 5000-row subset for speed |
| **Decision Tree** | max_depth ∈ {2,4,6,8,10,None} | `class_weight='balanced'` for imbalance |
| **LinearSVC** | C ∈ {0.01, 0.1, 1, 10} | Used over RBF SVC for O(n) speed on large data |
| **Random Forest** | n_estimators ∈ {50,100,200} | Best performer; feature importance plotted |
| **XGBoost** | scale_pos_weight=class_ratio | Future failure prediction variant |

### Clustering (Unsupervised Validation)
| Model | Config | Finding |
|---|---|---|
| **DBSCAN** | eps ∈ {0.5,1.0,1.5}, min_samples ∈ {5,10} | Noise cluster captured only ~1.4% of anomalies |
| **Spectral Clustering** | n_clusters=2, nearest_neighbors | Best cluster captured ~20.3% of anomalies |

> Both clustering models confirmed that unsupervised methods alone are insufficient for this task — supervised learning is necessary.

### Regression (Event Forecasting)
| Model | Target | Metric |
|---|---|---|
| **Linear Regression** | Next window's event count | RMSE, R² |
| **XGBoost Regressor** | Next window's event count | RMSE, R² (outperforms LR) |

---

## Tech Stack

| Tool | Why |
|---|---|
| **Pandas** | Ingesting 4M-row CSV, time-window grouping, feature engineering |
| **NumPy** | Metric averaging, array operations |
| **Scikit-learn** | All classification, clustering, regression models + metrics |
| **XGBoost** | Gradient boosted trees for both classification and regression — handles imbalanced data via `scale_pos_weight` |
| **Matplotlib + Seaborn** | Confusion matrices, ROC curves, time-series dashboards |
| **Kaggle Notebooks** | GPU/RAM environment for handling 4M-row dataset |

---

## Setup & Installation

### Run on Kaggle (Recommended — dataset is already there)

1. Go to [kaggle.com](https://www.kaggle.com) → Create Notebook
2. Add dataset: search `ayush2222/structured-bgl-logs-csv` → Add
3. Upload `system-monitoring-log-metric-analyser.ipynb`
4. Run All → all dependencies (sklearn, xgboost, seaborn) are pre-installed on Kaggle

### Run Locally

```bash
git clone https://github.com/your-username/system-monitoring-log-analyser.git
cd system-monitoring-log-analyser

pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter

# Download dataset from Kaggle manually:
# https://www.kaggle.com/datasets/ayush2222/structured-bgl-logs-csv
# Place the CSV at: /kaggle/input/datasets/ayush2222/structured-bgl-logs-csv/BGL.log_structured.csv
# Or update the path in Cell 2 of the notebook to your local path

jupyter notebook
```

---

## How to Run

Open `system-monitoring-log-metric-analyser.ipynb` → Run All Cells

The notebook runs in order:

| Section | What Happens |
|---|---|
| 1 | Load 4M rows, parse timestamps, drop redundant columns |
| 2 | 5-min windowing + feature extraction per node |
| 3 | Class imbalance analysis + binary label creation |
| 4 | Outlier detection (IQR) + winsorization |
| 5 | KNN with k-tuning + confusion matrix |
| 6 | Decision Tree with depth-tuning + tree visualization |
| 7 | LinearSVC with C-tuning |
| 8 | Random Forest with feature importance |
| 9 | ROC-AUC comparison across all 4 classifiers |
| 10 | DBSCAN + Spectral Clustering |
| 11 | Failure Prediction (RF + XGBoost classification) |
| 12 | Event Count Forecasting (Linear Reg + XGBoost Reg) |
| 13 | Final comparison table |
| 14 | System health monitoring dashboards |

---

## Results

> Exact numbers vary by run. Representative results from Kaggle:

### Anomaly Detection (Classification)

| Model | Accuracy | F1 (Anomaly) |
|---|---|---|
| KNN | ~99% | — |
| Decision Tree | ~99% | — |
| LinearSVC | ~99% | — |
| **Random Forest** | **~99%** | **Best** |
| XGBoost (Failure Pred.) | ~99% | Competitive |

> High accuracy is partially due to class imbalance (~90% normal). **F1 score on the anomaly class** is the real metric — Random Forest with `class_weight='balanced'` consistently leads.

### Clustering (Anomaly Capture Rate)

| Model | % of True Anomalies Captured |
|---|---|
| Spectral Clustering | ~20.3% |
| DBSCAN (noise cluster) | ~1.4% |

### Event Count Forecasting (Regression)

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | — | — |
| XGBoost Regressor | Lower RMSE | Higher R² |

---

## Screenshots

*Plots generated by the notebook — coming soon*

- ROC-AUC comparison (all 4 classifiers)
<img width="781" height="616" alt="image" src="https://github.com/user-attachments/assets/5587509c-5b66-4225-8025-eb20dcadc8ec" />

- Final model accuracy bar chart

<img width="805" height="416" alt="image" src="https://github.com/user-attachments/assets/9c379f96-e105-4a5c-b0f4-b48fcafbe39f" />

<img width="800" height="304" alt="image" src="https://github.com/user-attachments/assets/47949f50-9c56-47d1-a2d7-9ac320e2ce9a" />


---

## Key Findings

1. **Fatal ratio is the strongest feature** — the proportion of FATAL-level events in a time window is the most predictive signal for anomaly detection across all models.
2. **Clustering cannot replace supervised learning** — DBSCAN and Spectral Clustering captured under 20% of true anomalies, confirming that labeled supervised models are necessary for this task.
3. **Failure prediction works** — shifting labels one window forward allows the RF and XGBoost classifiers to predict upcoming failures before they occur, which is the most operationally valuable capability.
4. **XGBoost outperforms Linear Regression on event forecasting** — non-linear patterns in log bursts are better captured by gradient boosting.

---

## Author

**Vishwaraj Vikas Bhosale**  
B.Tech Computer Science & Engineering · PRN: 1032231758

---

## License

MIT License — free to use, modify, and share with attribution.
