# optimizer.py
import numpy as np
import matplotlib.pyplot as plt
from objective import calculate_fitness

def s_function(r, f=0.5, l=1.5):
    """Eq 19: Social interaction function"""
    return f * np.exp(-r / l) - np.exp(-r)

def run_goa(num_agents=20, max_iter=50, dim=3, lb=0.0, ub=3.0):
    print(f"Starting Grasshopper Optimization Algorithm (Agents: {num_agents}, Iterations: {max_iter})...")
    
    # Initialize population randomly within bounds
    positions = np.random.uniform(lb, ub, (num_agents, dim))
    fitness = np.zeros(num_agents)
    
    target_position = np.zeros(dim)
    target_fitness = float('inf')
    
    convergence_curve = np.zeros(max_iter)
    
    c_max = 1.0      # Section 3.3
    c_min = 0.00001  # Section 3.3
    
    # Initial fitness evaluation
    for i in range(num_agents):
        fitness[i] = calculate_fitness(positions[i])
        if fitness[i] < target_fitness:
            target_fitness = fitness[i]
            target_position = np.copy(positions[i])
            
    for l_iter in range(max_iter):
        # Eq 24: Adaptive parameter c
        c = c_max - l_iter * ((c_max - c_min) / max_iter)
        
        for i in range(num_agents):
            temp_position = np.zeros(dim)
            
            for j in range(num_agents):
                if i != j:
                    # Calculate distance
                    dist = np.linalg.norm(positions[j] - positions[i])
                    
                    # Algorithm step: Normalize distance between 1 and 4
                    normalized_dist = 1.0 + 3.0 * (dist / np.sqrt(dim * (ub - lb)**2))
                    if normalized_dist == 0:
                        normalized_dist = 1e-10
                        
                    # Eq 23: Position update calculation
                    s_val = s_function(normalized_dist)
                    direction = (positions[j] - positions[i]) / dist if dist != 0 else 0
                    temp_position += c * ((ub - lb) / 2.0) * s_val * direction
                    
            # Update final position with target component
            positions[i] = c * temp_position + target_position
            
            # Enforce boundaries
            positions[i] = np.clip(positions[i], lb, ub)
            
            # Evaluate new fitness
            current_fit = calculate_fitness(positions[i])
            if current_fit < target_fitness:
                target_fitness = current_fit
                target_position = np.copy(positions[i])
                
        convergence_curve[l_iter] = target_fitness
        print(f"Iteration {l_iter+1}/{max_iter} | Best RMSE: {target_fitness:.4f} | Best PID: {target_position}")

    # Plot Convergence (Matches Figure 7B)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_iter + 1), convergence_curve, color='blue', linewidth=2, label="GOA Convergence")
    plt.title("CHECKPOINT 7: GOA Convergence Curve (Matches Figure 7B)")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (RMSE)")
    plt.legend()
    plt.grid(True)
    
    save_path = "figures/convergence.png"
    plt.savefig(save_path)
    print(f"Convergence plot saved to '{save_path}'")
    
    return target_position, target_fitness