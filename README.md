# Parametric Quantum-Optimized Artificial Pancreas (PQO-AP)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![Framework: Qiskit](https://img.shields.io/badge/Quantum-Qiskit%201.0%2B-purple.svg)](https://qiskit.org/)
[![Simulator: SimGlucose](https://img.shields.io/badge/Simulator-UVA%2FPadova%20v0.2.1-orange.svg)](https://github.com/jinyungang/simglucose)

A closed-loop Type 1 Diabetes Mellitus (T1DM) automated insulin delivery (AID) simulation framework. This architecture integrates a non-linear 4-state Bergman minimal physiological plant, discrete Kalman filter state estimation, uncertainty-aware adaptive LQR-PID hybrid control, Quantum Approximate Optimization Algorithm (QAOA) discrete parameter tuning, and a model-predictive Low-Glucose Suspend (PLGS) safety layer with dynamic hysteresis.

---

## Executive Summary & Key Contributions

Type 1 Diabetes Mellitus (T1DM) is characterized by absolute insulin deficiency caused by autoimmune destruction of pancreatic beta cells. Closed-loop Automated Insulin Delivery (AID) systems (commonly referred to as an *Artificial Pancreas*) automate insulin infusion based on continuous glucose monitoring (CGM) telemetry. However, designing robust controllers remains challenging due to:
- **Subcutaneous Absorption Delays:** Insulin transit through subcutaneous tissues introduces delays of 30–60 minutes.
- **Unannounced Disturbance:** High-carbohydrate meals induce steep glucose rate-of-appearance spikes.
- **Sensor Noise & Artifacts:** Continuous Glucose Monitors (CGM) exhibit measurement noise ($\sigma_v^2 \approx 16\text{ mg}^2/\text{dL}^2$) and intrinsic physiological lag.
- **Hypoglycemia Risk:** Over-infusion during postprandial drop-offs can result in life-threatening neuroglycopenia ($G(t) < 54\text{ mg/dL}$).

The **PQO-AP** framework solves these challenges through a five-tier hybrid control pipeline:

1. **State Estimation:** A Discrete Kalman Filter (KF) reconstructs unobserved metabolic states—remote insulin action ($\hat{X}_k$) and plasma insulin concentration ($\hat{I}_k$)—from noisy interstitial CGM telemetry.
2. **Adaptive Hybrid Control:** A sigmoidally gated controller ($\alpha_k$) dynamically balances the nominal tracking precision of Discrete Algebraic Riccati Equation (DARE) based Linear Quadratic Regulation (LQR) with the rapid error correction of anti-windup Proportional-Integral-Derivative (PID) control.
3. **Quantum Parameter Tuning:** The controller gain configurations ($[K_p, K_i, K_d, R]$) are mapped into a 4-qubit Quadratic Unconstrained Binary Optimization (QUBO) landscape and optimized via multi-start QAOA ($p \in \{1, 2, 3\}$).
4. **Predictive Safety Suspension:** A 15-minute ($N=5$ step) discrete state-space predictor evaluates prospective glucose trajectories and activates Predictive Low-Glucose Suspend (PLGS) with dynamic hysteresis to zero out insulin delivery before hypoglycemic threshold breaches occur.
5. **In-Silico Verification:** Extensively benchmarked on the standardized FDA-accepted UVA/Padova virtual patient simulator (`simglucose`, 10 adults) under unannounced $70\text{g}$ carbohydrate meal challenges.

---

## System Architecture

```
                                 +-------------------------------------------------------+
                                 |                PQO-AP CLOSED-LOOP SYSTEM              |
                                 +-------------------------------------------------------+
                                                                 |
                                                                 v
+-----------------------+     Continuous CGM Telemetry     +-----------------------------+
|    T1DM Patient Plant | --------------------------------> | Discrete Kalman Filter (KF) |
|  (UVA/Padova / Bergman|                                   |  (State & Covariance Estim) |
+-----------------------+                                   +-----------------------------+
            ^                                                              |
            |                                 Reconstructed States [G_hat, X_hat, I_hat]
            | Subcutaneous                                                 v
            | Insulin Infusion                             +-----------------------------+
            | u[k] (U/min)                                 |  QAOA Gain Optimization     |
            |                                              |  (4-Qubit QUBO Formulation) |
            |                                              +-----------------------------+
            |                                                              |
            |                                 Optimized Tuning Vector [Kp, Ki, Kd, R]
            |                                                              v
+-----------------------+   Filtered Command u_cmd[k]      +-----------------------------+
|   PLGS Safety Layer   | <------------------------------- | Uncertainty-Aware Adaptive  |
| (15-min Predictor &   |                                  |     LQR-PID Hybrid Controller |
|  Dynamic Hysteresis)  |                                  +-----------------------------+
+-----------------------+
```

---

## Mathematical Formulation & Component Breakdown

### 1. Physiological Minimal Model (Bergman Plant)

The underlying glucose-insulin kinetics are governed by the modified 4-state non-linear Bergman Minimal Model:

$$\frac{dG(t)}{dt} = -p_1 \left( G(t) - G_b \right) - X(t) G(t) + D(t)$$

$$\frac{dX(t)}{dt} = -p_2 X(t) + p_3 \left( I(t) - I_b \right)$$

$$\frac{dI(t)}{dt} = -n \left( I(t) - I_b \right) + \frac{u(t)}{V_I}$$

$$\frac{dD(t)}{dt} = -A_{\text{meal}} D(t) + U_{\text{meal}}(t)$$

Where:
- $G(t)$: Blood Glucose concentration ($\text{mg/dL}$)
- $X(t)$: Remote insulin action in tissue compartment ($1/\text{min}$)
- $I(t)$: Plasma insulin concentration ($\mu\text{U/mL}$)
- $D(t)$: Gut absorption rate of meal carbohydrates ($\text{mg/dL/min}$)
- $G_b, I_b$: Basal glucose ($80.0\text{ mg/dL}$) and basal insulin ($7.0\text{ }\mu\text{U/mL}$)
- $p_1 = 0.028\text{ min}^{-1}$, $p_2 = 0.025\text{ min}^{-1}$, $p_3 = 1.3 \times 10^{-5}\text{ L/mU/min}^2$, $n = 0.092\text{ min}^{-1}$, $V_I = 12.0\text{ L}$.

### 2. Discrete-Time Kalman Filter

Given discrete sampling period $\Delta t = 3.0\text{ min}$, the linearized system state space $\mathbf{x}_k = [G_k - G_b, X_k, I_k - I_b, D_k]^T$ evolves as:

$$\mathbf{x}_{k+1} = \mathbf{A}_d \mathbf{x}_k + \mathbf{B}_d u_k + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$$

$$y_k = \mathbf{C}_d \mathbf{x}_k + v_k, \quad v_k \sim \mathcal{N}(0, R_{\text{cgm}})$$

The linear estimation cycle executes as follows:
- **Predict Step:**

$$\hat{\mathbf{x}}_{k|k-1} = \mathbf{A}_d \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B}_d u_{k-1}$$

$$\mathbf{P}_{k|k-1} = \mathbf{A}_d \mathbf{P}_{k-1|k-1} \mathbf{A}_d^T + \mathbf{Q}$$

- **Update Step:**

$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{C}_d^T \left( \mathbf{C}_d \mathbf{P}_{k|k-1} \mathbf{C}_d^T + R_{\text{cgm}} \right)^{-1}$$

$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \left( y_k - \mathbf{C}_d \hat{\mathbf{x}}_{k|k-1} \right)$$

$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{C}_d) \mathbf{P}_{k|k-1}$$

### 3. Uncertainty-Aware Adaptive LQR-PID Hybrid Controller

To dynamically balance stability and steady-state recovery, the control signal $u_{\text{raw}}[k]$ is synthesized via an estimation-uncertainty-gated convex combination:

$$u_{\text{LQR}}[k] = -\mathbf{K}_{\text{LQR}} \hat{\mathbf{x}}_{k|k} + u_{\text{basal}}$$

$$u_{\text{PID}}[k] = K_p e_k + K_i \sum_{j=0}^{k} e_j \Delta t + K_d \frac{e_k - e_{k-1}}{\Delta t}$$

$$\alpha_k = \frac{1}{1 + \exp\left( -\theta_{\text{adapt}} \cdot \left[ |e_k| - \lambda \cdot \text{Tr}(\mathbf{P}_{k|k}) \right] \right)}$$

$$u_{\text{raw}}[k] = \alpha_k u_{\text{PID}}[k] + (1 - \alpha_k) u_{\text{LQR}}[k]$$

Where $e_k = \hat{G}_k - G_{\text{target}}$ and $\text{Tr}(\mathbf{P}_{k|k})$ represents the trace of the Kalman estimation uncertainty.

### 4. QAOA Variational Quantum Parameter Tuning

The discrete parameter search space containing 16 candidate configurations of $[K_p, K_i, K_d, R]$ is mapped to a 4-qubit Hilbert space $\mathcal{H}_2^{\otimes 4}$.

The Problem Cost Hamiltonian $H_C$ encodes the performance index:

$$J = \sum_{k=1}^{N_T} \left( w_G (\hat{G}_k - G_{\text{ref}})^2 + w_u u_k^2 + w_{\Delta u} (\Delta u_k)^2 + w_{\text{hypo}} \max(0, 70 - \hat{G}_k)^2 \right)$$

$$H_C = \sum_{i,j=1}^{4} J_{ij} \sigma_z^{(i)} \sigma_z^{(j)} + \sum_{i=1}^{4} h_i \sigma_z^{(i)}$$

The variational QAOA state $|\gamma, \beta\rangle$ is generated using depth $p \in \{1, 2, 3\}$:

$$|\gamma, \beta\rangle = \prod_{l=1}^{p} e^{-i \beta_l H_B} e^{-i \gamma_l H_C} |+\rangle^{\otimes 4}$$

Where $H_B = \sum_{i=1}^{4} \sigma_x^{(i)}$ is the transverse mixer Hamiltonian. Variational angles $(\boldsymbol{\gamma}, \boldsymbol{\beta})$ are optimized classically via COBYLA across multiple random initializations.

### 5. Predictive Low-Glucose Suspend (PLGS) Safety Layer

Prior to commanding the pump, the 15-minute ($N_{\text{pred}}=5$ steps) future glucose concentration is predicted:

$$\hat{\mathbf{x}}_{k+N_{\text{pred}}} = \mathbf{A}_d^{N_{\text{pred}}} \hat{\mathbf{x}}_{k|k} + \sum_{j=0}^{N_{\text{pred}}-1} \mathbf{A}_d^{j} \mathbf{B}_d u_{k-1}$$

$$\hat{G}_{\text{pred}} = \mathbf{C}_d \hat{\mathbf{x}}_{k+N_{\text{pred}}} + G_b$$

The final pump command $u_{\text{pump}}[k]$ is subject to hard safety constraints:

$$u_{\text{pump}}[k] = \begin{cases} 0.0 & \text{if } \hat{G}_{k|k} < 88\text{ mg/dL} \text{ or } \hat{G}_{\text{pred}} < 78\text{ mg/dL} \\ 0.0 & \text{if PLGS latch active and } \hat{G}_{k|k} < 100\text{ mg/dL} \\ \min(u_{\text{max}}, \max(0, u_{\text{raw}}[k])) & \text{otherwise} \end{cases}$$

---

## Repository Structure

```text
artificial_pancreas/
├── PQO_AP/                            # Core Algorithmic Framework
│   ├── __init__.py
│   ├── safety/
│   │   ├── __init__.py
│   │   └── plgs.py                    # PLGS safety logic & trajectory predictor
│   └── experiments/
│       ├── __init__.py
│       ├── smoke_test_simglucose.py   # 24-hour single-patient verification (adult#001)
│       ├── run_conference_suite.py    # 48-hour multi-meal nominal evaluation suite
│       └── final_reviewer_benchmark.py# Complete 10-adult UVA/Padova benchmark & QAOA suite
├── conference/                        # Publication Artifacts
│   ├── figures/
│   │   ├── cohort_ablation_chart.png  # Statistical boxplots across 10 virtual adults
│   │   ├── qaoa_vs_classical.png      # QAOA expectation value convergence & histogram
│   │   └── simglucose_smoke_test.png  # 24-hour continuous glucose & insulin delivery curves
│   └── tables/
│       ├── final_quantum_vs_classical.csv # Multi-start QAOA (p=1..3) evaluation metrics
│       ├── patient_wise_pqo_ap.csv    # Individual performance breakdown (adult#001-010)
│       └── final_cohort_ablation.csv  # Full 6-controller ablation with Wilcoxon p-values
├── figures/                           # Supplemental Visualizations
│   ├── comparison.png                 # Bergman model baseline vs. adaptive closed-loop
│   └── kalman_tracking.png            # Kalman state estimation error trajectory
├── results/                           # Raw Experimental Output Logging
│   └── final_verified/
│       └── uva_padova_stress_10adults_raw.csv
├── ieee_artificial_pancreas_paper.html# Complete HTML format IEEE research paper
├── requirements.txt                   # Locked Python dependency manifest
└── README.md                          # Repository documentation
```

---

## Experimental Benchmark Results

### 1. Multi-Start QAOA vs. Classical Optimization (16-Candidate Discrete QUBO)

| Algorithm Variant | Circuit Depth ($p$) | COBYLA Runs | Best Index | Cost $J$ | Optimality Gap | Evaluations | Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exhaustive Grid Search** | -- | -- | 12 | 18.9598 | **0.00%** | 16 | 0.0010 |
| **Classical Random Search**| -- | -- | 12 | 18.9598 | **0.00%** | 250 | 0.0001 |
| **QAOA Multi-Start** | $p=1$ | 5 | 0 | 19.7195 | 4.01% | 250 | 0.1889 |
| **QAOA Multi-Start** | $p=2$ | 5 | 0 | 19.7195 | 4.01% | 250 | 0.1948 |
| **QAOA Multi-Start** | $p=3$ | 5 | 0 | 19.7195 | 4.01% | 250 | 0.1919 |

### 2. 10-Adult UVA/Padova Stress-Test Ablation (70g Unannounced Meal Challenges)

Evaluated across `adult#001` through `adult#010` over 24-hour continuous simulations ($1440$ steps per patient). Statistical significance was verified using the Wilcoxon signed-rank test against baseline PID.

| Controller Variant | Tracking RMSE (mg/dL) | Time in Range (70–180 mg/dL) | Time Below Range (<70 mg/dL) | Severe Hypo (<54 mg/dL) | Time Above Range (>180 mg/dL) | Mean Glucose (mg/dL) | Total Insulin (U) | PLGS Susp. (min) | Wilcoxon $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `C1_PID` (Baseline) | 46.54 ± 7.98 | 59.00% ± 11.11 | 40.66% ± 11.41 | 28.46% ± 11.55 | 0.33% ± 1.05 | 86.1 | 55.53 | 0.0 | Baseline |
| `C2_Pure_LQR` | 18.26 ± 4.15 | 100.00% ± 0.00 | 0.00% ± 0.00 | 0.00% ± 0.00 | 0.00% ± 0.00 | 123.1 | 35.31 | 0.0 | $p = 0.0020$ |
| `C3_Fixed_LQR_PID` | 17.35 ± 4.13 | 99.83% ± 0.53 | 0.17% ± 0.53 | 0.00% ± 0.00 | 0.00% ± 0.00 | 121.7 | 35.85 | 0.0 | $p = 0.0020$ |
| `C4_Unoptimized_Adaptive` | 15.95 ± 4.52 | 99.21% ± 1.68 | 0.79% ± 1.68 | 0.00% ± 0.00 | 0.00% ± 0.00 | 118.9 | 36.75 | 0.0 | $p = 0.0020$ |
| `C5_QAOA_Adaptive_No_PLGS` | 17.53 ± 4.10 | 99.85% ± 0.46 | 0.15% ± 0.46 | 0.00% ± 0.00 | 0.00% ± 0.00 | 122.1 | 35.70 | 0.0 | $p = 0.0020$ |
| **`C6_Full_PQO_AP` (Proposed)** | **17.53 ± 4.10** | **99.85% ± 0.46** | **0.15% ± 0.46** | **0.00% ± 0.00** | **0.00% ± 0.00** | **122.1** | **35.70** | **23.1** | **$p = 0.0020$** |

### 3. Patient-Wise Performance Breakdown (`C6_Full_PQO_AP`)

| Patient ID | Tracking RMSE (mg/dL) | TIR 70–180 (%) | TBR <70 (%) | Mean BG (mg/dL) | Peak BG (mg/dL) | Total Insulin (U) | PLGS Suspensions (min) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `adult#001` | 12.38 | 100.00% | 0.00% | 111.41 | 142.12 | 34.20 | 0.0 |
| `adult#002` | 21.05 | 100.00% | 0.00% | 127.30 | 165.41 | 35.10 | 27.0 |
| `adult#003` | 16.14 | 100.00% | 0.00% | 120.35 | 151.10 | 36.80 | 18.0 |
| `adult#004` | 23.41 | 98.54% | 1.46% | 130.12 | 172.90 | 37.50 | 45.0 |
| `adult#005` | 14.89 | 100.00% | 0.00% | 118.15 | 148.33 | 33.90 | 12.0 |
| `adult#006` | 19.32 | 100.00% | 0.00% | 124.60 | 159.80 | 36.10 | 21.0 |
| `adult#007` | 15.02 | 100.00% | 0.00% | 119.04 | 149.20 | 34.80 | 15.0 |
| `adult#008` | 21.80 | 100.00% | 0.00% | 128.90 | 167.10 | 38.20 | 33.0 |
| `adult#009` | 13.75 | 100.00% | 0.00% | 115.80 | 144.50 | 33.50 | 9.0 |
| `adult#010` | 17.54 | 100.00% | 0.00% | 122.30 | 156.40 | 36.90 | 51.0 |

---

## Installation & Setup Guide

### Environment Prerequisites
- Python 3.9, 3.10, or 3.11
- Operating System: Windows 10/11, macOS, or Linux (Ubuntu 20.04+)

### Step 1: Clone Repository
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/PQO-AP-Artificial-Pancreas.git
cd PQO-AP-Artificial-Pancreas
```

### Step 2: Virtual Environment Configuration

**On Windows (Command Prompt `cmd`):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**On Linux / macOS (Bash / Zsh):**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running Benchmarks & Simulations

### 1. Single Virtual Patient Verification (24-Hour Smoke Test)
Verifies closed-loop tracking and PLGS safety response on `adult#001` using the `simglucose` simulator:
```bash
python PQO_AP/experiments/smoke_test_simglucose.py
```

### 2. Full Conference Evaluation & QAOA Benchmark Suite
Executes multi-start QAOA parameter tuning ($p=1,2,3$) and generates nominal 48-hour comparison data:
```bash
python PQO_AP/experiments/run_conference_suite.py
```

### 3. Complete Reviewer Reproduction Suite (10-Adult Stress Test)
Runs the entire 10-adult cohort ablation study, computes Wilcoxon signed-rank $p$-values, and exports verified CSV summary tables:
```bash
python PQO_AP/experiments/final_reviewer_benchmark.py
```

---

## Medical Disclaimer

This software repository is an **in-silico computational research prototype** created solely for scientific evaluation and algorithmic benchmarking on mathematical models (Bergman minimal model and UVA/Padova T1DM simulator). 

- **Not a Medical Device:** This system has **not** been evaluated, certified, or cleared by the U.S. Food and Drug Administration (FDA), European Medicines Agency (EMA), or any other regulatory body.
- **Do Not Use for Human Infusion:** Do not connect this code to physical insulin pumps, continuous glucose monitors, or clinical patient management systems.
- **No Medical Advice:** Nothing within this repository constitutes clinical decision guidance, medical treatment recommendations, or therapeutic advice.

---

## Author Roster & Affiliation

All research, algorithmic implementation, and empirical evaluations were conducted by:

- **Y V S Harshith**
- **J Soven Reddy**
- **G Srikanth**
- **K Haveesh Chowdary**
- **Balasundhar C**
- **T Subeesh**

**Department of Artificial Intelligence**  
Amrita School of Artificial Intelligence, Coimbatore  
Amrita Vishwa Vidyapeetham, India

---

## Citation

If you utilize this framework, codebase, or benchmark methodology in your research, please cite our work:

```bibtex
@article{harshith2026pqoap,
  title={Quantum-Assisted Adaptive Control and Kalman-Based State Estimation for Closed-Loop Artificial Pancreas Systems},
  author={Harshith, Y V S and Reddy, J Soven and Srikanth, G and Chowdary, K Haveesh and C, Balasundhar and Subeesh, T},
  journal={Department of Artificial Intelligence, Amrita Vishwa Vidyapeetham},
  year={2026}
}
```