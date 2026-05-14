import numpy as np

class Trajectory:
    """
    Base class for defining a trajectory as a parametric function f(t).
    """
    def evaluate(self, t: float) -> np.ndarray:
        """
        Evaluate the parametric function at time t.
        Returns the desired pose f(t).
        """
        raise NotImplementedError("Subclasses must implement evaluate(t)")

    def cost(self, position: np.ndarray, t: float) -> float:
        """
        Compute the Euclidean distance between the given position and 
        the desired pose at time t.
        """
        desired_pose = self.evaluate(t)
        return float(np.linalg.norm(np.array(position) - desired_pose))


class CircleTrajectory(Trajectory):
    """
    A circular trajectory in the XY plane.
    """
    def __init__(self, center: np.ndarray, radius: float, angular_velocity: float = 1.0):
        self.center = np.array(center)
        self.radius = radius
        self.w = angular_velocity
        self.duration = 2 * np.pi / self.w

    def evaluate(self, t: float) -> np.ndarray:
        x_offset = self.radius * np.cos(self.w * t)
        y_offset = self.radius * np.sin(self.w * t)
        
        offset = np.zeros_like(self.center, dtype=float)
        offset[0] = x_offset
        offset[1] = y_offset
        
        return self.center + offset


class Figure8Trajectory(Trajectory):
    """
    A figure-8 (lemniscate) trajectory in the XY plane.
    """
    def __init__(self, center: np.ndarray, scale: float, angular_velocity: float = 1.0):
        self.center = np.array(center)
        self.scale = scale
        self.w = angular_velocity
        self.duration = 2 * np.pi / self.w

    def evaluate(self, t: float) -> np.ndarray:
        # Lemniscate of Gerono
        x_offset = self.scale * np.cos(self.w * t)
        y_offset = self.scale * np.sin(self.w * t) * np.cos(self.w * t)
        
        offset = np.zeros_like(self.center, dtype=float)
        offset[0] = x_offset
        offset[1] = y_offset
        
        return self.center + offset