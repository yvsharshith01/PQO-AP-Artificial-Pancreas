# controller.py
import parameters as p

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt, u_min, u_max):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        
        self.integral = 0.0
        self.prev_error = 0.0

    def calculate(self, current_glucose):
        # Eq 15: e(t) = G^ref(t) - G(t)
        error = p.Gb - current_glucose
        
        # PID Components
        P_out = self.Kp * error
        self.integral += error * self.dt
        I_out = self.Ki * self.integral
        derivative = (error - self.prev_error) / self.dt
        D_out = self.Kd * derivative
        
        # Raw Output
        raw_u = P_out + I_out + D_out
        
        # PHYSIOLOGICAL CORRECTION: Invert sign because process is reverse-acting.
        # If glucose is high (e.g., 150), error is -70. We need POSITIVE insulin.
        u = -raw_u 
        
        # Saturation Limits (0 to 10 limits from Figure 3)
        if u > self.u_max:
            u = self.u_max
            self.integral -= error * self.dt  # Anti-windup
        elif u < self.u_min:
            u = self.u_min
            self.integral -= error * self.dt  # Anti-windup
            
        self.prev_error = error
        return u