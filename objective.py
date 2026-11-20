# objective.py
import numpy as np
import parameters as p
from physiological_model import BergmanModel
from controller import PIDController

def calculate_fitness(gains):
    """
    gains: list or array [Kp, Ki, Kd]
    Returns the RMSE of the glucose tracking error.
    """
    Kp, Ki, Kd = gains
    
    # Penalize negative gains instantly without running simulation
    if Kp < 0 or Ki < 0 or Kd < 0:
        return 1e6

    model = BergmanModel()
    pid = PIDController(Kp, Ki, Kd, p.dt, p.u_min, p.u_max)
    
    state = np.array([p.Gb, 0.0, p.Ib, p.Gb])
    
    # Simulate a shorter 10-hour window (600 mins) for faster optimization
    # with a single meal at t=60
    sim_time_opt = 600
    times = np.arange(0, sim_time_opt, p.dt)
    glucose_history = np.zeros(len(times))
    
    for i, t in enumerate(times):
        glucose_history[i] = state[0]
        
        # Override meal disturbance for this isolated test
        D = p.B * np.exp(-p.d_rate * (t - 60)) if t >= 60 else 0.0
        
        # Custom step function to bypass the multi-meal setup in parameters.py
        dG_dt = -p.P1 * (state[0] - p.Gb) - state[1] * state[0] + D
        dX_dt = -p.P2 * state[1] + p.P3 * (state[2] - p.Ib)
        gamma_term = p.gamma * max(0.0, state[0] - p.h)
        u = pid.calculate(state[0])
        dI_dt = -p.n * (state[2] - p.Ib) - gamma_term + u
        dGsc_dt = (state[0] - state[3])/5.0 - p.Rutln
        
        derivatives = np.array([dG_dt, dX_dt, dI_dt, dGsc_dt])
        state = state + derivatives * p.dt # Simple Euler for speed during optimization
        
    # Calculate RMSE (Fitness)
    rmse = np.sqrt(np.mean((glucose_history - p.Gb)**2))
    
    # Heavy penalty if the patient goes into severe hypoglycemia (< 50 mg/dl)
    if np.min(glucose_history) < 50.0:
        rmse += 1000.0
        
    return rmse