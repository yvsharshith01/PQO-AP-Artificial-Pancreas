# clean_baseline_reproduction.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Ensure output directories exist
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# -------------------------------------------------------------
# 1. PARAMETERS & CONFIGURATION
# -------------------------------------------------------------
Gb = 80.0          # Target / basal glucose (mg/dL)
Ib = 7.0           # Basal insulin (uU/mL)
P1 = 0.0287        # Glucose effectiveness (1/min)
P2 = 0.0283        # Insulin action decay rate (1/min)
P3 = 5.03e-5       # Insulin sensitivity (1/min per uU/mL)
n = 0.0926         # Insulin clearance rate (1/min)
h = 80.0           # Threshold (mg/dL)
gamma = 0.0        # No endogenous insulin for T1DM
Rutln = 0.0

# Disturbances
B = 6.0            # Meal peak intensity
d_rate = 0.05      # Absorption decay rate (1/min)
meal_times = [400, 800, 1150, 1500, 1850, 2250, 2600]  # Multi-meal protocol

# Simulation limits
dt = 1.0           # Integration step (min)
sim_time = 2800    # Total duration (min) ~ 46 hours
u_min = 0.0        # Min insulin (uU/min)
u_max = 10.0       # Max insulin (uU/min)

# Paper published gains (Belmon & Auxillia 2020, Table 1)
paper_Kp = 0.4987
paper_Ki = 0.7985
paper_Kd = 0.6953

# -------------------------------------------------------------
# 2. PHYSIOLOGICAL SIMULATION ENGINE
# -------------------------------------------------------------
def meal_dist(t, custom_meals=None):
    meals = custom_meals if custom_meals is not None else meal_times
    d = 0.0
    for mt in meals:
        if t >= mt:
            d += B * np.exp(-d_rate * (t - mt))
    return d

def bergman_rk4_step(state, t, u, custom_meals=None):
    G, X, I, Gsc = state
    
    def derivatives(s, tau, inp):
        g, x, ins, gsc = s
        dist = meal_dist(tau, custom_meals)
        dG = -P1 * (g - Gb) - x * g + dist
        dX = -P2 * x + P3 * (ins - Ib)
        dI = -n * (ins - Ib) - gamma * max(0.0, g - h) + inp
        dGsc = (g - gsc) / 5.0 - Rutln
        return np.array([dG, dX, dI, dGsc])
    
    k1 = derivatives(state, t, u)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, u)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, u)
    k4 = derivatives(state + dt * k3, t + dt, u)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0
        
    def step(self, current_G):
        error = Gb - current_G
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        raw_u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # Physiological sign inversion: High glucose requires positive insulin
        u = -raw_u
        if u > u_max:
            u = u_max
            self.integral -= error * dt  # Anti-windup
        elif u < u_min:
            u = u_min
            self.integral -= error * dt
            
        self.prev_error = error
        return u

# -------------------------------------------------------------
# 3. GOA OPTIMIZER
# -------------------------------------------------------------
def s_force(r, f=0.5, l=1.5):
    return f * np.exp(-r / l) - np.exp(-r)

def evaluate_fitness(gains):
    Kp, Ki, Kd = gains
    if Kp < 0 or Ki < 0 or Kd < 0:
        return 1e6
        
    pid = PIDController(Kp, Ki, Kd)
    state = np.array([Gb, 0.0, Ib, Gb])
    opt_duration = 600  # 10-hour single-meal window for rapid evaluation
    times = np.arange(0, opt_duration, dt)
    glucose_profile = np.zeros(len(times))
    
    for idx, t in enumerate(times):
        glucose_profile[idx] = state[0]
        u = pid.step(state[0])
        state = bergman_rk4_step(state, t, u, custom_meals=[60])
        
    rmse = np.sqrt(np.mean((glucose_profile - Gb)**2))
    if np.min(glucose_profile) < 50.0:
        rmse += 1000.0  # Hypoglycemia safety penalty
    return rmse

def run_goa(num_agents=15, max_iter=30, bounds=(0.0, 5.0)):
    lb, ub = bounds
    dim = 3
    positions = np.random.uniform(lb, ub, (num_agents, dim))
    fitness = np.array([evaluate_fitness(pos) for pos in positions])
    
    best_idx = np.argmin(fitness)
    best_pos = positions[best_idx].copy()
    best_fit = fitness[best_idx]
    convergence = np.zeros(max_iter)
    
    c_max, c_min = 1.0, 0.00001
    for it in range(max_iter):
        c = c_max - it * ((c_max - c_min) / max_iter)
        for i in range(num_agents):
            temp_vec = np.zeros(dim)
            for j in range(num_agents):
                if i != j:
                    dist = np.linalg.norm(positions[j] - positions[i])
                    norm_dist = 1.0 + 3.0 * (dist / np.sqrt(dim * (ub - lb)**2))
                    norm_dist = max(norm_dist, 1e-10)
                    s = s_force(norm_dist)
                    direction = (positions[j] - positions[i]) / dist
                    temp_vec += c * ((ub - lb) / 2.0) * s * direction
            
            positions[i] = np.clip(c * temp_vec + best_pos, lb, ub)
            fit = evaluate_fitness(positions[i])
            if fit < best_fit:
                best_fit = fit
                best_pos = positions[i].copy()
        convergence[it] = best_fit
    return best_pos, best_fit, convergence

# -------------------------------------------------------------
# 4. EXECUTION & GENERATION OF THE 4 PLOTS
# -------------------------------------------------------------
print("Starting Clean Baseline Reproduction...")
np.random.seed(42)

times = np.arange(0, sim_time, dt)
hours = times / 60.0

# 1. Open-Loop Simulation
state_ol = np.array([Gb, 0.0, Ib, Gb])
g_ol = np.zeros(len(times))
for i, t in enumerate(times):
    g_ol[i] = state_ol[0]
    state_ol = bergman_rk4_step(state_ol, t, 0.0)

plt.figure(figsize=(9, 4.5))
plt.plot(hours, g_ol, 'r-', linewidth=2, label="Uncontrolled Glucose")
plt.axhline(Gb, color='blue', linestyle='--', label="Target Basal (80 mg/dL)")
plt.title("Plot 1: Open-Loop Glucose Response (Meal Disturbance, No Controller)")
plt.xlabel("Time (hr)")
plt.ylabel("Glucose (mg/dL)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("figures/open_loop.png")
plt.close()

# 2. Paper Baseline Simulation (Published Kp, Ki, Kd)
pid_paper = PIDController(paper_Kp, paper_Ki, paper_Kd)
state_paper = np.array([Gb, 0.0, Ib, Gb])
g_paper = np.zeros(len(times))
u_paper = np.zeros(len(times))
for i, t in enumerate(times):
    g_paper[i] = state_paper[0]
    u = pid_paper.step(state_paper[0])
    u_paper[i] = u
    state_paper = bergman_rk4_step(state_paper, t, u)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax1.plot(hours, g_paper, 'r:', linewidth=2, label="Blood Glucose (Published Gains)")
ax1.axhline(Gb, color='magenta', linestyle='-', label="Target (80 mg/dL)")
ax1.set_ylabel("Glucose (mg/dL)")
ax1.set_title("Plot 2: Published PID Gain Response (Overtuned Hypoglycemia Collapse)")
ax1.grid(True)
ax1.legend(loc="upper right")
ax2.plot(hours, u_paper, 'r-', linewidth=1.5, label="Insulin Injection")
ax2.set_ylabel("Insulin (uU/min)")
ax2.set_xlabel("Time (hr)")
ax2.set_ylim(-0.5, 11)
ax2.grid(True)
ax2.legend(loc="upper right")
plt.tight_layout()
plt.savefig("figures/controller_response.png")
plt.close()

# 3. GOA Optimization
opt_gains, opt_fitness, convergence_curve = run_goa(num_agents=15, max_iter=30)
opt_Kp, opt_Ki, opt_Kd = opt_gains

plt.figure(figsize=(8, 4.5))
plt.plot(range(1, 31), convergence_curve, 'b-', linewidth=2, label="Best Swarm Fitness")
plt.title("Plot 4: GOA Optimization Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Fitness (RMSE)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("figures/convergence.png")
plt.close()

# 4. Our GOA-Optimized Simulation
pid_opt = PIDController(opt_Kp, opt_Ki, opt_Kd)
state_opt = np.array([Gb, 0.0, Ib, Gb])
g_opt = np.zeros(len(times))
u_opt = np.zeros(len(times))
for i, t in enumerate(times):
    g_opt[i] = state_opt[0]
    u = pid_opt.step(state_opt[0])
    u_opt[i] = u
    state_opt = bergman_rk4_step(state_opt, t, u)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax1.plot(hours, g_opt, 'b:', linewidth=2, label="Blood Glucose (Our GOA-PID)")
ax1.axhline(Gb, color='magenta', linestyle='-', label="Target (80 mg/dL)")
ax1.set_ylabel("Glucose (mg/dL)")
ax1.set_title("Plot 3: Optimized GOA-PID Glucose Response (Stable Regulation)")
ax1.grid(True)
ax1.legend(loc="upper right")
ax2.plot(hours, u_opt, 'b-', linewidth=1.5, label="Insulin Injection")
ax2.set_ylabel("Insulin (uU/min)")
ax2.set_xlabel("Time (hr)")
ax2.set_ylim(-0.5, 11)
ax2.grid(True)
ax2.legend(loc="upper right")
plt.tight_layout()
plt.savefig("figures/optimized_response.png")
plt.close()

# -------------------------------------------------------------
# 5. COMPUTE ACCURATE METRICS & SAVE
# -------------------------------------------------------------
def calculate_metrics(g_trace, u_trace):
    e = Gb - g_trace
    rmse = float(np.sqrt(np.mean(e**2)))
    iae = float(np.sum(np.abs(e)) * dt)
    ise = float(np.sum(e**2) * dt)
    peak_g = float(np.max(g_trace))
    min_g = float(np.min(g_trace))
    return {"RMSE": rmse, "IAE": iae, "ISE": ise, "Peak_G": peak_g, "Min_G": min_g}

metrics_paper = calculate_metrics(g_paper, u_paper)
metrics_opt = calculate_metrics(g_opt, u_opt)

results_payload = {
    "optimal_Kp": float(opt_Kp),
    "optimal_Ki": float(opt_Ki),
    "optimal_Kd": float(opt_Kd),
    "objective_fitness": float(opt_fitness),
    "metrics_paper_gains": metrics_paper,
    "metrics_optimized": metrics_opt
}

with open("results/clean_results.json", "w") as f:
    json.dump(results_payload, f, indent=4)

print("\n--- CLEAN REPRODUCTION COMPLETE ---")
print(f"Optimized Kp: {opt_Kp:.4f}, Ki: {opt_Ki:.4f}, Kd: {opt_Kd:.4f}")
print(f"Final Objective Value (RMSE): {opt_fitness:.4f}")
print("All 4 plots saved in figures/")