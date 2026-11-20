# main.py
import numpy as np
import matplotlib.pyplot as plt
import parameters as p
from physiological_model import BergmanModel
from controller import PIDController
from optimizer import run_goa

def run_simulation(Kp, Ki, Kd):
    """Runs a 45-hour simulation and returns times, glucose, insulin, and RMSE"""
    model = BergmanModel()
    pid = PIDController(Kp, Ki, Kd, p.dt, p.u_min, p.u_max)
    
    state = np.array([p.Gb, 0.0, p.Ib, p.Gb])
    times = np.arange(0, p.sim_time, p.dt)
    
    glucose_history = np.zeros(len(times))
    u_history = np.zeros(len(times))
    
    for i, t in enumerate(times):
        glucose_history[i] = state[0]
        u = pid.calculate(state[0])
        u_history[i] = u
        state = model.step(state, t, u, p.dt)
        
    rmse = np.sqrt(np.mean((glucose_history - p.Gb)**2))
    
    # Penalize if it crashed into severe hypoglycemia
    if np.min(glucose_history) < 50.0:
        rmse += 1000.0
        
    return times / 60.0, glucose_history, u_history, rmse

if __name__ == "__main__":
    print("--- RUNNING CHECKPOINT 9: COMPARISON ---")
    
    # 1. Run Baseline (Paper's reported PSO-PID parameters)
    print("Simulating Baseline (Paper's PSO-PID parameters)...")
    Kp_base, Ki_base, Kd_base = 0.6052, 1.0248, 0.3231
    t_base, g_base, u_base, rmse_base = run_simulation(Kp_base, Ki_base, Kd_base)
    
    # 2. Run GOA Optimization for our assumed T1DM patient
    print("\nRunning GOA to find patient-specific parameters (30 iterations)...")
    best_gains, best_fitness = run_goa(num_agents=15, max_iter=30, dim=3, lb=0.0, ub=5.0)
    
    # 3. Run Optimized Controller
    print("\nSimulating Optimized GOA-PID...")
    t_opt, g_opt, u_opt, rmse_opt = run_simulation(best_gains[0], best_gains[1], best_gains[2])
    
    # 4. Plot Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Glucose Comparison
    ax1.plot(t_base, g_base, label=f"Paper Baseline (RMSE: {rmse_base:.2f})", color="red", linestyle="--")
    ax1.plot(t_opt, g_opt, label=f"Our GOA-PID (RMSE: {rmse_opt:.2f})", color="blue", linewidth=2)
    ax1.axhline(p.Gb, color='magenta', linestyle='-', label="Target (80 mg/dl)")
    ax1.set_ylabel("Glucose (mg/dl)")
    ax1.set_ylim(40, 160)
    ax1.set_title("CHECKPOINT 9: Controller Comparison")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    
    # Insulin Comparison
    ax2.plot(t_base, u_base, label="Baseline Insulin", color="red", linestyle="--", alpha=0.6)
    ax2.plot(t_opt, u_opt, label="GOA-PID Insulin", color="blue", alpha=0.8)
    ax2.set_xlabel("Time (hr)")
    ax2.set_ylabel("Insulin (μU/min)")
    ax2.set_ylim(-1, 11)
    ax2.legend(loc="upper right")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("figures/comparison.png")
    print("\nSuccess. Comparison saved to 'figures/comparison.png'.")