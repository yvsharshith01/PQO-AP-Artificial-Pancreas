# adaptive_qaoa_controller.py
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import parameters as p
from state_space_kalman import A_d, B_d, C_d, GlucoseKalmanFilter
from physiological_model import BergmanModel

# -------------------------------------------------------------
# 1. DISCRETE LQR CONTROLLER SOLVER
# -------------------------------------------------------------
def solve_dlqr(A, B, Q, R):
    """Solves the Discrete Algebraic Riccati Equation (DARE) and returns gain K."""
    P_mat = scipy.linalg.solve_discrete_are(A, B, Q, R)
    # K = (R + B^T P B)^(-1) B^T P A
    K = np.linalg.inv(R + B.T @ P_mat @ B) @ (B.T @ P_mat @ A)
    return K, P_mat

# -------------------------------------------------------------
# 2. ADAPTIVE LQR-PID CONTROLLER CLASS
# -------------------------------------------------------------
class AdaptiveLQRPIDController:
    def __init__(self, Kp, Ki, Kd, Q_diag, R_val, alpha_0=0.5, a1=0.02, a2=0.01):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
        self.Q = np.diag(Q_diag)
        self.R = np.array([[R_val]])
        self.K_lqr, _ = solve_dlqr(A_d, B_d, self.Q, self.R)
        
        self.alpha_0 = alpha_0
        self.a1 = a1  # Sensitivity to glucose tracking error
        self.a2 = a2  # Sensitivity to Kalman state uncertainty
        
        self.integral = 0.0
        self.prev_error = 0.0
        
    def step(self, x_hat_delta, current_G, uncertainty_trace):
        error = p.Gb - current_G
        
        # 1. PID Component
        self.integral += error * p.dt
        derivative = (error - self.prev_error) / p.dt
        raw_pid = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        u_pid = -raw_pid # Physiological correction (positive insulin for high glucose)
        
        # 2. LQR Component: u_LQR = -K * [delta_G, delta_X, delta_I]^T
        u_lqr = (-self.K_lqr @ x_hat_delta).item()
        
        # 3. Dynamic Adaptive Blending Alpha
        z = self.alpha_0 + self.a1 * abs(error) - self.a2 * uncertainty_trace
        alpha = float(1.0 / (1.0 + np.exp(-z)))
        alpha = np.clip(alpha, 0.05, 0.95)
        
        # Blended Control Output
        u_blended = alpha * u_lqr + (1.0 - alpha) * u_pid
        
        # Saturation & Anti-windup
        u_final = float(np.clip(u_blended, p.u_min, p.u_max))
        if u_blended > p.u_max or u_blended < p.u_min:
            self.integral -= error * p.dt
            
        self.prev_error = error
        return u_final, alpha, u_lqr, u_pid

# -------------------------------------------------------------
# 3. QUBO & QAOA PARAMETER OPTIMIZATION
# -------------------------------------------------------------
PARAM_SPACE = [
    # Kp,    Ki,     Kd,     R_val
    [0.10,  0.015,  0.75,   1.0],
    [0.12,  0.020,  0.80,   2.0],
    [0.15,  0.025,  0.85,   1.5],
    [0.18,  0.030,  0.90,   0.5],
    [0.11,  0.018,  0.82,   1.2],
    [0.14,  0.022,  0.88,   2.5],
    [0.16,  0.028,  0.70,   0.8],
    [0.20,  0.035,  0.95,   3.0],
    [0.09,  0.012,  0.65,   1.0],
    [0.13,  0.021,  0.84,   1.8],
    [0.15,  0.024,  0.88,   2.2],
    [0.17,  0.026,  0.78,   1.1],
    [0.19,  0.032,  0.92,   0.7],
    [0.22,  0.040,  1.00,   4.0],
    [0.08,  0.010,  0.60,   0.9],
    [0.25,  0.050,  1.10,   5.0],
]

def simulate_candidate(candidate):
    """Evaluates an objective score for candidate parameters on a single-meal run with noise."""
    Kp, Ki, Kd, R_val = candidate
    ctrl = AdaptiveLQRPIDController(
        Kp=Kp, Ki=Ki, Kd=Kd,
        Q_diag=[10.0, 1.0, 0.1],
        R_val=R_val,
        alpha_0=0.5
    )
    kf = GlucoseKalmanFilter(
        Q_cov=np.diag([1e-2, 1e-6, 1e-3]),
        R_cov=np.array([[4.0]])
    )
    
    sim_mins = 600
    times = np.arange(0, sim_mins, p.dt)
    state = np.array([p.Gb, 0.0, p.Ib, p.Gb])
    g_hist = np.zeros(len(times))
    u_hist = np.zeros(len(times))
    
    np.random.seed(101)
    for idx, t in enumerate(times):
        g_hist[idx] = state[0]
        y_meas = state[0] + np.random.normal(0, 2.0)
        
        kf.predict(u_hist[idx-1] if idx > 0 else 0.0)
        est_full, P_cov = kf.update(y_meas)
        x_hat_delta = kf.x_hat
        
        u, _, _, _ = ctrl.step(x_hat_delta, est_full[0], np.trace(P_cov))
        u_hist[idx] = u
        
        meal = p.B * np.exp(-p.d_rate * (t - 60)) if t >= 60 else 0.0
        dG = -p.P1 * (state[0] - p.Gb) - state[1] * state[0] + meal
        dX = -p.P2 * state[1] + p.P3 * (state[2] - p.Ib)
        dI = -p.n * (state[2] - p.Ib) + u
        dGsc = (state[0] - state[3]) / 5.0
        state = state + np.array([dG, dX, dI, dGsc]) * p.dt
        
    rmse = np.sqrt(np.mean((g_hist - p.Gb)**2))
    hypo_penalty = 500.0 if np.min(g_hist) < 65.0 else 0.0
    smoothness = 0.05 * np.sum(np.diff(u_hist)**2)
    return rmse + hypo_penalty + smoothness

def build_qubo_cost_vector():
    """Evaluates all 16 discrete states to form the diagonal QUBO cost Hamiltonian."""
    costs = np.array([simulate_candidate(c) for c in PARAM_SPACE])
    norm_costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs) + 1e-8)
    return costs, norm_costs

def run_qaoa_optimization(p_depth=2, n_evals=40):
    """Simulates a depth-p QAOA circuit optimizing expectation value <psi|H_C|psi>."""
    print("Building QUBO Hamiltonian & executing QAOA parameter optimization...")
    raw_costs, norm_costs = build_qubo_cost_vector()
    n_states = len(PARAM_SPACE)
    
    psi_0 = np.ones(n_states) / np.sqrt(n_states)
    H_C = norm_costs
    
    best_expected_energy = float('inf')
    best_gamma, best_beta = None, None
    best_probs = None
    history_energy = []
    
    np.random.seed(42)
    for step in range(n_evals):
        gamma = np.random.uniform(0, 2 * np.pi, p_depth)
        beta = np.random.uniform(0, np.pi, p_depth)
        
        state_vec = psi_0.copy()
        for layer in range(p_depth):
            state_vec = state_vec * np.exp(-1j * gamma[layer] * H_C)
            mix_matrix = np.cos(beta[layer]) * np.eye(n_states) - 1j * np.sin(beta[layer]) * np.fliplr(np.eye(n_states))
            state_vec = mix_matrix @ state_vec
            
        probs = np.abs(state_vec)**2
        probs /= np.sum(probs)
        
        expected_cost = np.sum(probs * raw_costs)
        history_energy.append(expected_cost)
        
        if expected_cost < best_expected_energy:
            best_expected_energy = expected_cost
            best_gamma, best_beta = gamma, beta
            best_probs = probs
            
    optimal_idx = int(np.argmax(best_probs))
    optimal_params = PARAM_SPACE[optimal_idx]
    
    print(f"QAOA Optimization converged to Candidate #{optimal_idx}")
    print(f"Optimal Gains: Kp={optimal_params[0]}, Ki={optimal_params[1]}, Kd={optimal_params[2]} | LQR R={optimal_params[3]}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(history_energy, 'b-o', markersize=4, linewidth=1.5)
    ax1.set_title("QAOA Variational Energy Convergence")
    ax1.set_xlabel("Classical Optimizer Step")
    ax1.set_ylabel("Expected Cost <H_C>")
    ax1.grid(True)
    
    ax2.bar(range(n_states), best_probs, color='teal', alpha=0.8)
    ax2.set_title("Optimal QAOA Measurement State Probabilities")
    ax2.set_xlabel("Discrete Candidate Index")
    ax2.set_ylabel("Probability")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("figures/qaoa_optimization_landscape.png")
    plt.close()
    
    return optimal_params

# -------------------------------------------------------------
# 4. FULL 45-HOUR SIMULATION WITH DYNAMIC ALPHA BLENDING
# -------------------------------------------------------------
def run_full_adaptive_simulation(opt_params):
    print("Running 45-Hour Closed-Loop Simulation with Adaptive LQR-PID...")
    Kp, Ki, Kd, R_val = opt_params
    
    controller = AdaptiveLQRPIDController(
        Kp=Kp, Ki=Ki, Kd=Kd,
        Q_diag=[10.0, 1.0, 0.1],
        R_val=R_val,
        alpha_0=0.5
    )
    kf = GlucoseKalmanFilter(
        Q_cov=np.diag([1e-2, 1e-6, 1e-3]),
        R_cov=np.array([[4.0]])
    )
    model = BergmanModel()
    
    times = np.arange(0, p.sim_time, p.dt)
    hours = times / 60.0
    
    state = np.array([p.Gb, 0.0, p.Ib, p.Gb])
    g_hist = np.zeros(len(times))
    noisy_cgm = np.zeros(len(times))
    u_hist = np.zeros(len(times))
    alpha_hist = np.zeros(len(times))
    u_lqr_hist = np.zeros(len(times))
    u_pid_hist = np.zeros(len(times))
    
    np.random.seed(42)
    for idx, t in enumerate(times):
        g_hist[idx] = state[0]
        y_meas = state[0] + np.random.normal(0, 2.0)
        noisy_cgm[idx] = y_meas
        
        kf.predict(u_hist[idx-1] if idx > 0 else 0.0)
        est_full, P_cov = kf.update(y_meas)
        x_hat_delta = kf.x_hat
        
        u, alpha, u_lqr, u_pid = controller.step(x_hat_delta, est_full[0], np.trace(P_cov))
        u_hist[idx] = u
        alpha_hist[idx] = alpha
        u_lqr_hist[idx] = u_lqr
        u_pid_hist[idx] = u_pid
        
        state = model.step(state, t, u, p.dt)
        
    rmse = np.sqrt(np.mean((g_hist - p.Gb)**2))
    print(f"Adaptive LQR-PID 45-hr Simulation complete. RMSE: {rmse:.4f}")
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(hours, noisy_cgm, '.', color='gray', alpha=0.3, label="Noisy CGM")
    axes[0].plot(hours, g_hist, 'b-', linewidth=2, label="Adaptive LQR-PID Glucose")
    axes[0].axhline(p.Gb, color='magenta', linestyle='--', label="Target (80 mg/dL)")
    axes[0].set_ylabel("Glucose (mg/dL)")
    axes[0].set_title(f"Adaptive LQR-PID Regulation (RMSE: {rmse:.2f} mg/dL)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)
    
    axes[1].plot(hours, alpha_hist, 'purple', linewidth=1.8, label="Blending Coefficient (alpha)")
    axes[1].axhline(0.5, color='gray', linestyle=':')
    axes[1].set_ylabel("Alpha")
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="upper right")
    axes[1].grid(True)
    
    axes[2].plot(hours, u_hist, 'black', linewidth=1.5, label="Final Blended Insulin u(t)")
    axes[2].plot(hours, u_lqr_hist, 'green', linestyle='--', alpha=0.6, label="u_LQR component")
    axes[2].plot(hours, u_pid_hist, 'orange', linestyle=':', alpha=0.6, label="u_PID component")
    axes[2].set_xlabel("Time (hr)")
    axes[2].set_ylabel("Insulin (uU/min)")
    axes[2].set_ylim(-1, 11)
    axes[2].legend(loc="upper right")
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("figures/adaptive_blending_response.png")
    plt.close()
    print("Adaptive response plot saved to 'figures/adaptive_blending_response.png'")
    
    return opt_params, rmse

if __name__ == "__main__":
    best_params = run_qaoa_optimization(p_depth=2, n_evals=40)
    run_full_adaptive_simulation(best_params)