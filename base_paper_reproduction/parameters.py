# parameters.py

# --- ASSUMED STANDARD BERGMAN PARAMETERS FOR T1DM ---
P1 = 0.0287        
P2 = 0.0283        
P3 = 5.03e-5       
n = 0.0926         
h = 80.0           
gamma = 0.0        
Gb = 80.0          
Ib = 7.0           
Rutln = 0.0        

# --- DISTURBANCE / MEAL PARAMETERS ---
B = 6.0            
d_rate = 0.05      
# Multiple meals to replicate the 45-hour peaks in Figure 6
meal_times = [400, 800, 1150, 1500, 1850, 2250, 2600] 

# --- SIMULATION PARAMETERS ---
dt = 1.0           
sim_time = 2800    # ~46 hours in minutes to match paper's x-axis

# --- BASE PAPER CONTROLLER PARAMETERS (Table 1: GOA-PID) ---
Kp_goa = 0.4987
Ki_goa = 0.7985
Kd_goa = 0.6953
u_min = 0.0        # Figure 3: Insulin Low Limit
u_max = 10.0       # Figure 3: Insulin High Limit