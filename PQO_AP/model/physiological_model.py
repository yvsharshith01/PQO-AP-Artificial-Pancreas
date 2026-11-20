# physiological_model.py
import numpy as np
import parameters as p

class BergmanModel:
    def __init__(self):
        pass

    def meal_disturbance(self, t):
        """Eq 14: D(t) = B * exp(-d_rate * t) applied to multiple meals"""
        D = 0.0
        for m_time in p.meal_times:
            if t >= m_time:
                D += p.B * np.exp(-p.d_rate * (t - m_time))
        return D

    def derivatives(self, state, t, u):
        G, X, I, Gsc = state
        D = self.meal_disturbance(t)

        # Eq 10: dG/dt = -P1[G - Gb] - X*G + D(t)
        dG_dt = -p.P1 * (G - p.Gb) - X * G + D
        
        # Eq 11: dX/dt = -P2*X + P3[I - Ib]
        dX_dt = -p.P2 * X + p.P3 * (I - p.Ib)
        
        # Eq 12: dI/dt = -n[I - Ib] - gamma[G - h] + U(t)
        # Note: gamma term is floored at 0 (only active if G > h)
        gamma_term = p.gamma * max(0.0, G - p.h)
        dI_dt = -p.n * (I - p.Ib) - gamma_term + u
        
        # Eq 13: dGsc/dt = (G - Gsc)/5 - Rutln
        dGsc_dt = (G - Gsc)/5.0 - p.Rutln

        return np.array([dG_dt, dX_dt, dI_dt, dGsc_dt])

    def step(self, state, t, u, dt):
        """4th-Order Runge-Kutta numerical integration"""
        k1 = self.derivatives(state, t, u)
        k2 = self.derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, u)
        k3 = self.derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, u)
        k4 = self.derivatives(state + dt * k3, t + dt, u)
        
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return new_state