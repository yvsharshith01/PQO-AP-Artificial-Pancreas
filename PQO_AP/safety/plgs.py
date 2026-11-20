# PQO_AP/safety/plgs.py
import numpy as np

class PLGSSafetyFilter:
    """
    Model-Based Predictive Low-Glucose Suspend (PLGS) with Hysteresis.
    Predicts future glucose over a horizon H using the discrete state-space
    transition matrix A_d under zero future insulin (u = 0).
    """
    def __init__(
        self,
        A_d,
        G_basal,
        suspend_thresh=85.0,
        pred_suspend_thresh=75.0,
        resume_thresh=90.0,
        pred_resume_thresh=80.0,
        horizon_steps=5
    ):
        self.A_d = A_d
        self.G_basal = G_basal
        self.suspend_thresh = suspend_thresh
        self.pred_suspend_thresh = pred_suspend_thresh
        self.resume_thresh = resume_thresh
        self.pred_resume_thresh = pred_resume_thresh
        self.horizon_steps = horizon_steps

        # State & telemetry tracking
        self.is_suspended = False
        self.activation_count = 0
        self.suspended_steps = 0
        self.history_suspended = []
        self.history_predicted_g = []

    def predict_zero_insulin_trajectory(self, x_hat_delta):
        """Propagates state perturbation x_hat_delta forward assuming u = 0."""
        x_curr = x_hat_delta.reshape((3, 1)).copy()
        min_pred_g = float(x_curr[0, 0] + self.G_basal)
        for _ in range(self.horizon_steps):
            x_curr = self.A_d @ x_curr
            pred_g = float(x_curr[0, 0] + self.G_basal)
            if pred_g < min_pred_g:
                min_pred_g = pred_g
        return min_pred_g

    def filter(self, u_command, x_hat_delta, current_est_g):
        min_pred_g = self.predict_zero_insulin_trajectory(x_hat_delta)
        self.history_predicted_g.append(min_pred_g)

        if not self.is_suspended:
            # Check suspend triggers
            if (current_est_g < self.suspend_thresh) or (min_pred_g < self.pred_suspend_thresh):
                self.is_suspended = True
                self.activation_count += 1
        else:
            # Hysteresis: require both current and predicted glucose to clear resume thresholds
            if (current_est_g > self.resume_thresh) and (min_pred_g > self.pred_resume_thresh):
                self.is_suspended = False

        if self.is_suspended:
            self.suspended_steps += 1
            self.history_suspended.append(1)
            return 0.0
        else:
            self.history_suspended.append(0)
            return u_command

    def get_telemetry(self, dt_minutes):
        return {
            "plgs_activations": self.activation_count,
            "suspended_minutes": self.suspended_steps * dt_minutes,
            "suspended_steps": self.suspended_steps
        }