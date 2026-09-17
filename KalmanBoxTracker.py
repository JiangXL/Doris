import os
import numpy as np

class KalmanBoxTracker:
    """Constant-velocity Kalman filter tracking one fin box (SORT-style).

    State: [cx, cy, s, r, vx, vy, vs] where (cx, cy) is the box center,
    s the box area and r the aspect ratio w/h. The velocity terms absorb
    consistent frame-to-frame displacement from camera shake and dolphin
    motion.
    """

    def __init__(self, box):
        self.x = np.zeros((7, 1))
        self.x[:4] = self._box_to_z(box)
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0  # high initial velocity uncertainty
        self.Q = np.eye(7) * 0.01
        self.Q[6, 6] *= 0.01
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.time_since_update = 0

    @staticmethod
    def _box_to_z(box):
        w = box[2] - box[0]
        h = box[3] - box[1]
        return np.array([box[0] + w / 2, box[1] + h / 2,
                         w * h, w / float(h)]).reshape((4, 1))

    @staticmethod
    def _x_to_box(x):
        cx, cy = x[0, 0], x[1, 0]
        s = max(x[2, 0], 1.0)
        w = np.sqrt(s * x[3, 0])
        h = s / w
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def predict(self):
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self._x_to_box(self.x)

    def update(self, box):
        self.time_since_update = 0
        y = self._box_to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
