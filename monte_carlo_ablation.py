# monte_carlo_ablation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parameters as p
from state_space_kalman import A_d, B_d, GlucoseKalmanFilter
from adaptive_qaoa_controller import AdaptiveLQRPIDController, solve_dlqr
from controller import PIDController

N_TRIALS = 100

def run_trial(controller_type, seed):
    np.random.seed(seed)
    
    p3_perturbed = p.P3 * np.random.uniform(0.8, 1.2)
    b_perturbed = p.B * np.random.uniform(0.75, 1.25)
    noise_sigma = np.random.uniform(1.5, 3.0)
    
    dt = 1.0
    sim_time = 1440
    times = np.arange(0, sim_time, dt)
    state = np.array([p.Gb, 0.0, p.Ib, p.Gb])
    
    if controller_type == "PID_Fixed":
        ctrl = PIDController(Kp=0.4987, Ki=0.7985, Kd=0.6953, dt=dt, u_min=p.u_min, u_max=p.u_max)
    elif controller_type == "GOA_PID":
        ctrl = PIDController(Kp=0.1512, Ki=0.0248, Kd=0.8841, dt=dt, u_min=p.u_min, u_max=p.u_max)
    elif controller_type == "Pure_LQR":
        K_lqr, _ = solve_dlqr(A_d, B_d, np.diag([10.0, 1.0, 0.1]), np.array([[1.5]]))
    elif controller_type == "Adaptive_LQR_PID":
        adaptive_ctrl = AdaptiveLQRPIDController(Kp=0.14, Ki=0.022, Kd=0.88, Q_diag=[10.0, 1.0, 0.1], R_val=1.5)
        kf = GlucoseKalmanFilter(Q_cov=np.diag([1e-2, 1e-6, 1e-3]), R_cov=np.array([[noise_sigma**2]]))
        
    g_trace = np.zeros(len(times))
    u_trace = np.zeros(len(times))
    
    meals = [300, 700, 1100]
    
    for idx, t in enumerate(times):
        g_trace[idx] = state[0]
        y_noisy = state[0] + np.random.normal(0, noise_sigma)
        
        if controller_type in ["PID_Fixed", "GOA_PID"]:
            u = ctrl.calculate(y_noisy)
        elif controller_type == "Pure_LQR":
            delta_x = np.array([[y_noisy - p.Gb], [0.0], [0.0]])
            raw_lqr = (-K_lqr @ delta_x).item()
            u = float(np.clip(raw_lqr, p.u_min, p.u_max))
        elif controller_type == "Adaptive_LQR_PID":
            kf.predict(u_trace[idx-1] if idx > 0 else 0.0)
            est_full, P_cov = kf.update(y_noisy)
            u, _, _, _ = adaptive_ctrl.step(kf.x_hat, est_full[0], np.trace(P_cov))
            
        u_trace[idx] = u
        
        dist = 0.0
        for mt in meals:
            if t >= mt:
                dist += b_perturbed * np.exp(-p.d_rate * (t - mt))
                
        dG = -p.P1 * (state[0] - p.Gb) - state[1] * state[0] + dist
        dX = -p.P2 * state[1] + p3_perturbed * (state[2] - p.Ib)
        dI = -p.n * (state[2] - p.Ib) + u
        dGsc = (state[0] - state[3]) / 5.0
        state = state + np.array([dG, dX, dI, dGsc]) * dt
        
    rmse = np.sqrt(np.mean((g_trace - p.Gb)**2))
    mae = np.mean(np.abs(g_trace - p.Gb))
    tir = np.mean((g_trace >= 70.0) & (g_trace <= 180.0)) * 100.0
    hypo = np.mean(g_trace < 70.0) * 100.0
    severe_hypo = np.mean(g_trace < 54.0) * 100.0
    insulin_tot = np.sum(u_trace) * (dt / 60.0)
    smoothness = np.sum(np.diff(u_trace)**2)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "TIR": tir,
        "Hypo": hypo,
        "Severe_Hypo": severe_hypo,
        "Insulin_Total": insulin_tot,
        "Smoothness": smoothness,
        "Glucose_Trace": g_trace
    }

def run_monte_carlo_study():
    print(f"Running Monte Carlo Robustness Evaluation ({N_TRIALS} Stochastic Trials)...")
    controllers = ["PID_Fixed", "GOA_PID", "Pure_LQR", "Adaptive_LQR_PID"]
    
    results_summary = []
    traces = {c: [] for c in controllers}
    
    for c_name in controllers:
        print(f"Evaluating: {c_name}...")
        c_metrics = []
        for trial in range(N_TRIALS):
            m = run_trial(c_name, seed=1000 + trial)
            traces[c_name].append(m["Glucose_Trace"])
            c_metrics.append({k: v for k, v in m.items() if k != "Glucose_Trace"})
            
        df = pd.DataFrame(c_metrics)
        results_summary.append({
            "Controller": c_name,
            "RMSE_Mean": df["RMSE"].mean(),
            "RMSE_Std": df["RMSE"].std(),
            "MAE_Mean": df["MAE"].mean(),
            "TIR_Mean (%)": df["TIR"].mean(),
            "Hypo_Mean (%)": df["Hypo"].mean(),
            "Severe_Hypo (%)": df["Severe_Hypo"].mean(),
            "Insulin_Usage": df["Insulin_Total"].mean(),
            "Control_Smoothness": df["Smoothness"].mean()
        })
        
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("results/final_research_metrics.csv", index=False)
    print("\n--- MONTE CARLO RESULTS SUMMARY ---")
    print(summary_df.to_string(index=False))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    hours = np.arange(0, 1440, 1.0) / 60.0
    flat_axes = axes.flatten()
    
    colors = {"PID_Fixed": "red", "GOA_PID": "orange", "Pure_LQR": "green", "Adaptive_LQR_PID": "blue"}
    
    for i, c_name in enumerate(controllers):
        arr = np.array(traces[c_name])
        mean_curve = np.mean(arr, axis=0)
        std_curve = np.std(arr, axis=0)
        
        flat_axes[i].plot(hours, mean_curve, color=colors[c_name], linewidth=2, label="Mean Glucose")
        flat_axes[i].fill_between(hours, mean_curve - std_curve, mean_curve + std_curve, color=colors[c_name], alpha=0.2, label="+/- 1 Std Dev")
        flat_axes[i].axhline(p.Gb, color='black', linestyle=':', label="Target (80)")
        flat_axes[i].axhline(70.0, color='red', linestyle='--', alpha=0.5, label="Hypo Threshold (70)")
        flat_axes[i].set_title(f"{c_name}")
        flat_axes[i].set_ylabel("Glucose (mg/dL)")
        flat_axes[i].grid(True)
        if i >= 2:
            flat_axes[i].set_xlabel("Time (hr)")
        flat_axes[i].legend(loc="upper right", fontsize=8)
        
    plt.tight_layout()
    plt.savefig("figures/monte_carlo_distribution.png")
    plt.close()
    print("Monte Carlo confidence bands saved to 'figures/monte_carlo_distribution.png'")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    names = summary_df["Controller"]
    
    ax1.bar(names, summary_df["RMSE_Mean"], yerr=summary_df["RMSE_Std"], capsize=5, color=['salmon', 'orange', 'lightgreen', 'cornflowerblue'])
    ax1.set_title("Tracking Error (RMSE)")
    ax1.set_ylabel("RMSE (mg/dL)")
    ax1.grid(True, axis='y')
    ax1.tick_params(axis='x', rotation=15)
    
    ax2.bar(names, summary_df["Hypo_Mean (%)"], color=['red', 'orange', 'green', 'blue'])
    ax2.set_title("Hypoglycemia Exposure (< 70 mg/dL)")
    ax2.set_ylabel("Time (%)")
    ax2.grid(True, axis='y')
    ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig("figures/ablation_comparison.png")
    plt.close()
    print("Ablation comparison saved to 'figures/ablation_comparison.png'")

if __name__ == "__main__":
    run_monte_carlo_study()