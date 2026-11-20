# state_space_kalman.py
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import parameters as p
from physiological_model import BergmanModel

# -------------------------------------------------------------
# 1. ANALYTICAL LINEARIZATION AROUND BASAL EQUILIBRIUM
# -------------------------------------------------------------
# Equilibrium: G* = Gb = 80, X* = 0, I* = Ib = 7, u* = 0, D* = 0
# State vector: x = [delta_G, delta_X, delta_I]^T
#
# d(delta_G)/dt = -P1*delta_G - G* * delta_X - X* * delta_G + delta_D
#               = -P1*delta_G - Gb*delta_X + delta_D
# d(delta_X)/dt = -P2*delta_X + P3*delta_I
# d(delta_I)/dt = -n*delta_I + delta_u

A_c = np.array([
    [-p.P1,      -p.Gb,       0.0],
    [ 0.0,       -p.P2,       p.P3],
    [ 0.0,        0.0,       -p.n ]
])

B_c = np.array([
    [0.0],
    [0.0],
    [1.0]
])

E_c = np.array([
    [1.0],
    [0.0],
    [0.0]
])

C_c = np.array([[1.0, 0.0, 0.0]]) # Only blood glucose is measured by CGM

# Discretization via matrix exponential (zero-order hold with dt = 1.0 min)
dt = p.dt
n_states = 3

M = np.zeros((n_states + 1, n_states + 1))
M[:n_states, :n_states] = A_c
M[:n_states, n_states:] = B_c
M_exp = scipy.linalg.expm(M * dt)

A_d = M_exp[:n_states, :n_states]
B_d = M_exp[:n_states, n_states:]
C_d = C_c.copy()

# -------------------------------------------------------------
# 2. DISCRETE-TIME KALMAN FILTER CLASS
# -------------------------------------------------------------
class GlucoseKalmanFilter:
    def __init__(self, Q_cov, R_cov, x0=None, P0=None):
        self.A = A_d
        self.B = B_d
        self.C = C_d
        
        self.Q = Q_cov   # Process noise covariance (3x3)
        self.R = R_cov   # Measurement noise covariance (1x1)
        
        # State: [delta_G, delta_X, delta_I]
        self.x_hat = np.zeros((3, 1)) if x0 is None else x0.reshape((3, 1))
        self.P = np.eye(3) * 10.0 if P0 is None else P0

    def predict(self, u_control):
        # Time update (Prediction)
        u_val = np.array([[u_control]])
        self.x_hat = self.A @ self.x_hat + self.B @ u_val
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x_hat, self.P

    def update(self, y_measured_glucose):
        # Measurement innovation: y_k - C*x_hat (relative to baseline Gb)
        delta_y = np.array([[y_measured_glucose - p.Gb]])
        y_pred = self.C @ self.x_hat
        innovation = delta_y - y_pred
        
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        
        # State update
        self.x_hat = self.x_hat + K @ innovation
        self.P = (np.eye(3) - K @ self.C) @ self.P
        
        # Full physiological state: [G, X, I]
        full_state = np.array([
            self.x_hat[0, 0] + p.Gb,
            self.x_hat[1, 0],
            self.x_hat[2, 0] + p.Ib
        ])
        return full_state, self.P

# -------------------------------------------------------------
# 3. VERIFICATION RUN
# -------------------------------------------------------------
def test_kalman_filter():
    print("Testing Linearization & Kalman Filter under Noisy CGM Sensor...")
    np.random.seed(42)
    
    model = BergmanModel()
    
    # Process & measurement noise covariances
    Q_w = np.diag([1e-2, 1e-6, 1e-3])
    R_v = np.array([[4.0]]) # CGM noise std dev = 2.0 mg/dL (variance = 4.0)
    
    kf = GlucoseKalmanFilter(Q_cov=Q_w, R_cov=R_v)
    
    sim_time = 1200 # 20 hours
    times = np.arange(0, sim_time, dt)
    
    true_states = np.zeros((len(times), 4))
    estimated_states = np.zeros((len(times), 3))
    noisy_cgm = np.zeros(len(times))
    covariance_trace = np.zeros(len(times))
    
    state = np.array([p.Gb, 0.0, p.Ib, p.Gb])
    
    # Simple basal infusion with single meal at t=120
    for i, t in enumerate(times):
        true_states[i] = state
        
        # Inject realistic Gaussian sensor noise to glucose measurement
        sensor_noise = np.random.normal(0, np.sqrt(R_v[0, 0]))
        measured_g = state[0] + sensor_noise
        noisy_cgm[i] = measured_g
        
        # Kalman step (open-loop baseline u=0)
        kf.predict(u_control=0.0)
        est_full, P_mat = kf.update(measured_g)
        
        estimated_states[i] = est_full
        covariance_trace[i] = np.trace(P_mat)
        
        # Meal at t=120
        dist = p.B * np.exp(-p.d_rate * (t - 120)) if t >= 120 else 0.0
        
        # RK4 step
        dG = -p.P1 * (state[0] - p.Gb) - state[1] * state[0] + dist
        dX = -p.P2 * state[1] + p.P3 * (state[2] - p.Ib)
        dI = -p.n * (state[2] - p.Ib)
        dGsc = (state[0] - state[3]) / 5.0
        derivatives = np.array([dG, dX, dI, dGsc])
        state = state + derivatives * dt
        
    # Plotting verification
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    hours = times / 60.0
    
    # Glucose tracking
    axes[0].plot(hours, noisy_cgm, '.', color='gray', alpha=0.4, label="Noisy CGM Sensor")
    axes[0].plot(hours, true_states[:, 0], 'r-', linewidth=2, label="True Glucose G(t)")
    axes[0].plot(hours, estimated_states[:, 0], 'b--', linewidth=2, label="Kalman Estimate G_hat(t)")
    axes[0].set_ylabel("Glucose (mg/dL)")
    axes[0].set_title("Kalman Filter Verification: State Estimation Under Sensor Noise")
    axes[0].legend(loc="upper right")
    axes[0].grid(True)
    
    # Latent Insulin Action X(t)
    axes[1].plot(hours, true_states[:, 1], 'r-', linewidth=2, label="True Insulin Action X(t)")
    axes[1].plot(hours, estimated_states[:, 1], 'b--', linewidth=2, label="Kalman Estimate X_hat(t)")
    axes[1].set_ylabel("Action X (1/min)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)
    
    # Plasma Insulin I(t)
    axes[2].plot(hours, true_states[:, 2], 'r-', linewidth=2, label="True Insulin I(t)")
    axes[2].plot(hours, estimated_states[:, 2], 'b--', linewidth=2, label="Kalman Estimate I_hat(t)")
    axes[2].set_xlabel("Time (hr)")
    axes[2].set_ylabel("Insulin I (uU/mL)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("figures/kalman_tracking.png")
    print("Kalman verification complete. Plot saved to 'figures/kalman_tracking.png'")
    
    # Print numerical tracking accuracy of hidden states
    x_rmse = np.sqrt(np.mean((true_states[:, 1] - estimated_states[:, 1])**2))
    i_rmse = np.sqrt(np.mean((true_states[:, 2] - estimated_states[:, 2])**2))
    print(f"Estimation RMSE -> Latent Action X: {x_rmse:.6f} | Plasma Insulin I: {i_rmse:.4f}")

if __name__ == "__main__":
    test_kalman_filter()