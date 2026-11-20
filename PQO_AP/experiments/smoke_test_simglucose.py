# PQO_AP/experiments/smoke_test_simglucose.py
import os
import sys
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from datetime import datetime

from simglucose.simulation.env import T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario import CustomScenario
from simglucose.controller.base import Action

# Add project root to path so we can import PLGS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from PQO_AP.safety.plgs import PLGSSafetyFilter

os.makedirs("results/benchmark", exist_ok=True)
os.makedirs("conference/figures", exist_ok=True)

# -------------------------------------------------------------
# 1. DISCRETE STATE-SPACE & KALMAN FILTER (dt = 3.0 min for Dexcom)
# -------------------------------------------------------------
DT = 3.0
TARGET_BG = 110.0  # Clinical euglycemic target (mg/dL)

# Continuous Bergman parameters scaled for U/min pump actuation
P1 = 0.0287
P2 = 0.0283
P3 = 5.03e-5
N_CLEAR = 0.0926
GAIN_SCALE = 100.0  # Maps U/min pump command to plasma insulin appearance

A_c = np.array([
    [-P1,  -TARGET_BG,  0.0],
    [0.0,  -P2,         P3],
    [0.0,   0.0,       -N_CLEAR]
])
B_c = np.array([[0.0], [0.0], [GAIN_SCALE]])
C_d = np.array([[1.0, 0.0, 0.0]])

M = np.zeros((4, 4))
M[:3, :3] = A_c
M[:3, 3:] = B_c
M_exp = scipy.linalg.expm(M * DT)
A_d = M_exp[:3, :3]
B_d = M_exp[:3, 3:]

def solve_dlqr(A, B, Q, R):
    P_mat = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P_mat @ B) @ (B.T @ P_mat @ A)
    return K

class SimGlucoseKalmanFilter:
    def __init__(self, Q_cov, R_cov):
        self.Q = Q_cov
        self.R = R_cov
        self.x_hat = np.zeros((3, 1))
        self.P = np.diag([25.0, 1e-4, 1.0])

    def predict(self, u_net):
        self.x_hat = A_d @ self.x_hat + B_d * u_net
        self.P = A_d @ self.P @ A_d.T + self.Q

    def update(self, cgm_bg):
        y_delta = np.array([[cgm_bg - TARGET_BG]])
        innov = y_delta - (C_d @ self.x_hat)
        S = C_d @ self.P @ C_d.T + self.R
        K = self.P @ C_d.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ innov
        self.P = (np.eye(3) - K @ C_d) @ self.P
        return float(self.x_hat[0, 0] + TARGET_BG), self.P

# -------------------------------------------------------------
# 2. CONTROLLERS FOR SIMGLUCOSE (OUTPUT IN U/min)
# -------------------------------------------------------------
class BaselinePIDAdapter:
    def __init__(self, Kp=0.0006, Ki=0.00001, Kd=0.015, u_basal=0.015):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.u_basal = u_basal
        self.integral = 0.0
        self.prev_err = 0.0

    def step(self, cgm_bg):
        err = cgm_bg - TARGET_BG  # Positive when hyperglycemic -> positive insulin
        self.integral += err * DT
        deriv = (err - self.prev_err) / DT
        u = self.u_basal + self.Kp * err + self.Ki * self.integral + self.Kd * deriv
        u_clamped = float(np.clip(u, 0.0, 0.15))
        if u != u_clamped:
            self.integral -= err * DT
        self.prev_err = err
        return u_clamped, 0.0

class PQOAPAdapter:
    def __init__(self, u_basal=0.015):
        self.u_basal = u_basal
        self.Kp = 0.0007
        self.Ki = 0.00001
        self.Kd = 0.018
        self.integral = 0.0
        self.prev_err = 0.0

        self.K_lqr = solve_dlqr(A_d, B_d, np.diag([1.0, 10.0, 0.01]), np.array([[500.0]]))
        self.kf = SimGlucoseKalmanFilter(
            Q_cov=np.diag([2.0, 1e-6, 0.05]),
            R_cov=np.array([[15.0]])  # Dexcom sensor noise variance
        )
        self.plgs = PLGSSafetyFilter(
            A_d=A_d,
            G_basal=TARGET_BG,
            suspend_thresh=85.0,
            pred_suspend_thresh=75.0,
            resume_thresh=90.0,
            pred_resume_thresh=80.0,
            horizon_steps=5  # 5 * 3 min = 15-min prediction horizon
        )
        self.last_u_net = 0.0

    def step(self, cgm_bg):
        # 1. Kalman State Estimation
        self.kf.predict(self.last_u_net)
        est_g, P_cov = self.kf.update(cgm_bg)
        trace_p = float(np.trace(P_cov))

        # 2. PID Component (using filtered glucose)
        err = est_g - TARGET_BG
        self.integral += err * DT
        deriv = (err - self.prev_err) / DT
        u_pid = self.u_basal + self.Kp * err + self.Ki * self.integral + self.Kd * deriv
        self.prev_err = err

        # 3. LQR Component
        u_lqr = self.u_basal + float((-self.K_lqr @ self.kf.x_hat).item())

        # 4. Uncertainty-Aware Sigmoid Blending
        z = 0.2 + 0.03 * abs(err) - 0.02 * trace_p
        alpha = float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.1, 0.9))
        u_blended = float(np.clip(alpha * u_lqr + (1.0 - alpha) * u_pid, 0.0, 0.15))

        # 5. Model-Based PLGS Safety Filter
        u_safe = self.plgs.filter(u_blended, self.kf.x_hat, est_g)

        if u_safe != u_blended or u_blended in (0.0, 0.15):
            self.integral -= err * DT  # Anti-windup during suspension/saturation

        self.last_u_net = u_safe - self.u_basal
        return u_safe, alpha

# -------------------------------------------------------------
# 3. RUN SINGLE-PATIENT SMOKE TEST (adult#001, 24 HOURS)
# -------------------------------------------------------------
def run_patient_sim(mode="PQO_AP"):
    start_time = datetime(2026, 11, 19, 0, 0, 0)
    # 3 standard daily meals: 45g at 7h, 70g at 12h, 80g at 18h
    meals = [(7 * 60, 45), (12 * 60, 70), (18 * 60, 80)]
    scenario = CustomScenario(start_time=start_time, scenario=meals)

    patient = T1DPatient.withName("adult#001")
    sensor = CGMSensor.withName("Dexcom", seed=42)
    pump = InsulinPump.withName("Insulet")
    env = T1DSimEnv(patient, sensor, pump, scenario)

    obs, _, _, _ = env.reset()
    controller = PQOAPAdapter() if mode == "PQO_AP" else BaselinePIDAdapter()

    steps = int(24 * 60 / DT)  # 480 steps = 24 hours
    bg_true, bg_cgm, insulin_cmd, alphas = [], [], [], []

    for _ in range(steps):
        cgm_val = float(obs.CGM)
        true_val = float(env.patient.observation.Gsub)
        u_rate, alpha = controller.step(cgm_val)

        bg_cgm.append(cgm_val)
        bg_true.append(true_val)
        insulin_cmd.append(u_rate)
        alphas.append(alpha)

        obs, _, done, _ = env.step(Action(basal=u_rate, bolus=0.0))
        if done:
            break

    bg_arr = np.array(bg_true)
    u_arr = np.array(insulin_cmd)

    metrics = {
        "Controller": mode,
        "Mean_BG": np.mean(bg_arr),
        "RMSE_110": np.sqrt(np.mean((bg_arr - TARGET_BG) ** 2)),
        "TIR_70_180 (%)": np.mean((bg_arr >= 70) & (bg_arr <= 180)) * 100.0,
        "TBR_lt_70 (%)": np.mean(bg_arr < 70) * 100.0,
        "TBR_lt_54 (%)": np.mean(bg_arr < 54) * 100.0,
        "TAR_gt_180 (%)": np.mean(bg_arr > 180) * 100.0,
        "TAR_gt_250 (%)": np.mean(bg_arr > 250) * 100.0,
        "Total_Insulin_U": np.sum(u_arr) * DT,
        "Jitter_Vu": np.sum(np.diff(u_arr) ** 2),
        "PLGS_Activations": controller.plgs.activation_count if mode == "PQO_AP" else 0,
        "Suspended_Mins": controller.plgs.suspended_steps * DT if mode == "PQO_AP" else 0.0,
    }
    return bg_arr, np.array(bg_cgm), u_arr, np.array(alphas), metrics, controller

if __name__ == "__main__":
    print("Running SimGlucose (UVA/Padova adult#001) 24-Hour Smoke Test...")
    bg_pid, _, u_pid, _, m_pid, _ = run_patient_sim("Baseline_PID")
    bg_pqo, cgm_pqo, u_pqo, alpha_pqo, m_pqo, pqo_ctrl = run_patient_sim("PQO_AP")

    print("\n--- SINGLE-PATIENT SMOKE TEST RESULTS (adult#001) ---")
    for k in m_pid:
        v_pid = f"{m_pid[k]:.3f}" if isinstance(m_pid[k], float) else str(m_pid[k])
        v_pqo = f"{m_pqo[k]:.3f}" if isinstance(m_pqo[k], float) else str(m_pqo[k])
        print(f"{k:20s} | PID: {v_pid:12s} | PQO-AP: {v_pqo}")

    # Plot closed-loop glucose + PLGS suspension shading
    hours = np.arange(len(bg_pqo)) * DT / 60.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    ax1.axhspan(70, 180, color="green", alpha=0.08, label="Target Range (70–180 mg/dL)")
    ax1.plot(hours, bg_pid, "r--", linewidth=1.6, label="Baseline PID")
    ax1.plot(hours, bg_pqo, "b-", linewidth=2.0, label="PQO-AP (Kalman + Adaptive LQR-PID + PLGS)")
    ax1.axhline(110, color="black", linestyle=":", label="Target Set Point (110 mg/dL)")
    ax1.set_ylabel("Blood Glucose (mg/dL)")
    ax1.set_title("UVA/Padova (SimGlucose adult#001) 24-Hour Closed-Loop Verification")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True)

    susp = np.array(pqo_ctrl.plgs.history_suspended)
    ax2.plot(hours, u_pid, "r--", alpha=0.6, label="Baseline PID Pump (U/min)")
    ax2.plot(hours, u_pqo, "b-", linewidth=1.6, label="PQO-AP Safe Pump (U/min)")
    ax2.fill_between(hours, 0, 0.15, where=(susp == 1), color="orange", alpha=0.25, label="PLGS Insulin Suspension")
    ax2.set_ylabel("Insulin Rate (U/min)")
    ax2.set_xlabel("Time (Hours)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("conference/figures/simglucose_smoke_test.png")
    print("\nSaved figure to 'conference/figures/simglucose_smoke_test.png'")