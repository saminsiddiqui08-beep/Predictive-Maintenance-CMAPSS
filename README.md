\# Predictive Maintenance of Turbofan Engines using Explainable AI (XAI)



\## Project Overview

This repository contains a complete deep learning pipeline for predicting the Remaining Useful Life (RUL) of aerospace turbofan engines. The project transitions from a "Black Box" predictive model to a "Glass Box" Explainable AI architecture, ensuring that safety-critical predictions are thermodynamically transparent and mathematically defensible.



\## Dataset

This project utilizes the \*\*NASA CMAPSS FD004\*\* dataset. FD004 is the most complex dataset in the CMAPSS benchmark, featuring:

\* 6 distinct operational conditions

\* 2 simultaneous failure modes (HPC Degradation \& Fan Degradation)

\* Massive end-of-life sensor noise and vibration



\## Architecture \& Results

The pipeline utilizes a strict engine-level isolation split (80/20) to prevent temporal data leakage and ensure honest evaluation metrics.



1\. \*\*Baseline Model:\*\* Advanced BiLSTM

&#x20;  \* \*\*Test RMSE:\*\* 16.71 cycles

&#x20;  \* \*Status:\* Highly accurate but uninterpretable (Black Box).

2\. \*\*Glass-Box Model:\*\* Temporal Attention BiLSTM + SHAP

&#x20;  \* \*\*Test RMSE:\*\* 19.59 cycles

&#x20;  \* \*Status:\* Physically auditable. The model trades a marginal 2.8 cycles of accuracy to unlock cycle-level temporal attention matrices and SHAP-based feature attribution, allowing engineers to verify the physical logic behind every prediction.



\## Execution Pipeline

To reproduce the methodology, run the Jupyter Notebooks in the following order:

1\. `01\_Data\_Exploration.ipynb`: Generates sliding windows and physical sensor masks.

2\. `02\_Model\_Training.ipynb`: Trains the Advanced BiLSTM and establishes the safety baseline.

3\. `03\_XAI\_Extraction.ipynb`: Trains the Attention BiLSTM and extracts `.npy` arrays for game-theory validation.

4\. `04\_Visualization\_and\_Validation.ipynb`: Calculates the NASA PHMS08 Asymmetric Score and generates the Thermodynamic Attention Overlays.

