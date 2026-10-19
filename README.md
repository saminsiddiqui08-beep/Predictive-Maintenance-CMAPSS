# Predictive Maintenance of Turbofan Engines using Explainable AI

Predicting the Remaining Useful Life (RUL) of turbofan engines with a BiLSTM model, and making each prediction auditable using temporal attention and SHAP.

This repository contains the code for the paper "Temporal Attention and Shapley Additive Explanations for the Prediction of Remaining Useful Life of Turbofan Engines", submitted to the 2026 AIAA Region VII Student Conference.

## Why this project
A model that predicts RUL accurately is not much use in aerospace if nobody can check why it made the prediction. This project trains two models on the same data: one standard BiLSTM, and one with a temporal attention layer whose weights can be inspected directly. SHAP is then applied to the attention model to attribute each prediction to individual sensors. The aim is to test whether the model's reasoning lines up with known engine physics.

## Dataset
NASA CMAPSS **FD004**, the hardest of the four subsets:
* 6 operating conditions
* 2 fault modes (HPC degradation and fan degradation)
* 249 training engines, 248 test engines

**The data is not included in this repository.** Download the Turbofan Engine Degradation Simulation Data Set from the NASA Prognostics Data Repository, then place `train_FD004.txt`, `test_FD004.txt`, and `RUL_FD004.txt` in a `Data/` folder at the project root.

> Saxena, A. and Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set*. NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.

## Preprocessing
* 7 sensors dropped because they carry little degradation information (they track flight state rather than component wear), leaving 14 sensors + 3 operational settings = 17 features
* K-Means clusters the 3 operational settings into 6 flight regimes, and each regime is normalized with its own MinMaxScaler. The scalers are fit on all 249 training engines before the train/validation split, so validation values influence the scaling; the test set is only transformed
* 30-cycle sliding windows; RUL labels clipped at 125 cycles for both training and test, following the standard piecewise-linear convention for CMAPSS
* The 80/20 split is done at the engine level, so no engine's windows appear in both training and validation

## Models and results

| Model | Test RMSE | PHM08 Score | Interpretability |
|---|---|---|---|
| Advanced BiLSTM (final-state BiLSTM in the paper) | 16.71 cycles | 2225 | Not analyzed (SHAP was applied only to the attention model) |
| Attention BiLSTM + SHAP | 19.59 cycles | 3355 | Temporal attention + SHAP attribution |

Both models use the same hidden size, dropout, Huber loss, weight decay, gradient clipping, and 60-epoch schedule, and the only architectural difference is the attention layer. However, the results come from a single training run, the two models started from different initial weights, and their final validation errors were close (mean per-batch MSE of 166.68 vs. 172.06). The 2.88-cycle test gap should therefore not be attributed to the attention layer alone.

The attention model produces per-cycle attention weights and per-sensor SHAP attributions. SHAP ranks S11 (HPC Static Pressure) as the strongest driver of RUL predictions, consistent with the HPC fault mode, and the attention overlay is plotted against that same sensor. The attention weights, however, concentrate on the final few cycles of each window rather than following changes in S11.

> **Note on Reproducibility:** Reported results were obtained using GPU (CUDA) training. PyTorch does not guarantee bit-for-bit identical results between CPU and GPU execution, even with a fixed random seed, due to differences in underlying floating-point math libraries (cuDNN vs. MKL/oneDNN). Re-running these notebooks on CPU only hardware may therefore produce different RMSE values. All numbers in this README and in the paper come from the GPU run.

## Setup
    pip install -r requirements.txt

Developed and tested on Python 3.11.15. Package versions are pinned in `requirements.txt`.

## Running the notebooks
Run in order. Each notebook saves `.npy` files that the next one loads:

1. `01_Data_Exploration.ipynb` — preprocessing, regime clustering, normalization, sliding windows
2. `02_Model_Training.ipynb` — trains the baseline and the Advanced BiLSTM
3. `03_XAI_Extraction.ipynb` — trains the Attention BiLSTM, extracts attention weights and SHAP values
4. `04_Visualization_and_Validation.ipynb` — NASA PHM08 scoring, figures, and the literature comparison table

> **Note on the literature table:** The table in notebook 04 lists FD001 results for two of the earlier studies and includes one row that could not be verified. The paper's Table 1 uses verified FD004 results only.

## License
MIT License. See `LICENSE`.
