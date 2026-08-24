# Predictive Maintenance of Turbofan Engines using Explainable AI

Predicting the Remaining Useful Life (RUL) of turbofan engines with a BiLSTM model, and making each prediction auditable using temporal attention and SHAP.

## Why this project
A model that predicts RUL accurately is not much use in aerospace if nobody can check why it made the prediction. This project trains two models on the same data: one standard BiLSTM, and one with a temporal attention layer whose weights can be inspected directly. SHAP is then applied to the attention model to attribute each prediction to individual sensors. The aim is to test whether the model's reasoning lines up with known engine physics.

## Dataset
NASA CMAPSS **FD004**, the hardest of the four subsets:
* 6 operating conditions
* 2 concurrent fault modes (HPC degradation and fan degradation)
* 249 training engines, 248 test engines
* Heavy sensor noise near end of life

**The data is not included in this repository.** Download the Turbofan Engine Degradation Simulation Data Set from the NASA Prognostics Data Repository, then place `train_FD004.txt`, `test_FD004.txt`, and `RUL_FD004.txt` in a `Data/` folder at the project root.

> Saxena, A. and Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set*. NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

## Preprocessing
* 7 sensors dropped for near-zero variance (they track flight state rather than component wear), leaving 14 sensors + 3 operational settings = 17 features
* K-Means clusters the 3 operational settings into 6 flight regimes, and each regime is normalized with its own MinMaxScaler
* 30-cycle sliding windows; RUL labels clipped at 125 cycles, following the standard piecewise-linear convention for CMAPSS
* The 80/20 split is done at the engine level, so no engine's data appears in both training and validation

## Models and results

| Model | Test RMSE | NASA Score | Interpretability |
|---|---|---|---|
| Advanced BiLSTM | 16.71 cycles | 2225 | None (black box) |
| Attention BiLSTM + SHAP | 19.59 cycles | 3355 | Temporal attention + SHAP attribution |

Both models use the same hidden size, dropout, Huber loss, weight decay, gradient clipping, and 60-epoch schedule. The only architectural difference is the attention layer, so the 2.88-cycle gap can be attributed to attention itself rather than to other changes.

In exchange for that accuracy, the attention model produces per-cycle attention weights and per-sensor SHAP attributions. SHAP ranks S11 (HPC Static Pressure) as the strongest driver of RUL predictions, and the attention overlay is plotted against that same sensor.

> **Note on Reproducibility:** Reported results were obtained using GPU (CUDA) training. PyTorch does not guarantee bit-for-bit identical results between CPU and GPU execution, even with a fixed random seed, due to differences in underlying floating-point math libraries (cuDNN vs. MKL/oneDNN). Re-running these notebooks on CPU only hardware may therefore produce slightly different RMSE values. Qualitative findings, including the SHAP sensor importance ranking, remain consistent across hardware.

## Setup
    pip install -r requirements.txt

Developed and tested on Python 3.11.15. Package versions are pinned in `requirements.txt`.

## Running the notebooks
Run in order. Each notebook saves `.npy` files that the next one loads:

1. `01_Data_Exploration.ipynb` — preprocessing, regime clustering, normalization, sliding windows
2. `02_Model_Training.ipynb` — trains the baseline and the Advanced BiLSTM
3. `03_XAI_Extraction.ipynb` — trains the Attention BiLSTM, extracts attention weights and SHAP values
4. `04_Visualization_and_Validation.ipynb` — NASA PHM08 scoring, figures, and the literature comparison table
